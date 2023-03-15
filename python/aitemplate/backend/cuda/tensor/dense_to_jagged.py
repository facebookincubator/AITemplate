#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
The back-end bindings of the dense_to_jagged op.
"""
from typing import Any, Dict, List, Optional

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm, IntVar, JaggedIntVar, Tensor
from aitemplate.utils import shape_utils


CONSTANT_TEMPLATE = jinja2.Template(
    """
#define FUSED_ELE_THREAD_SIZE 256

const int N_ELEMENTS_PER_THREAD = sizeof({{read_t}}) / sizeof({{data_t}});
    """
)

KERNEL_COMPUTE_DENSE_IDX_THEN_JAGGED_IDX_TEMPLATE = jinja2.Template(
    """
  // first compute the dense_idx from the blockIdx and threadIdx
  const {{index_type}} dense_idx = blockIdx.x * FUSED_ELE_THREAD_SIZE + threadIdx.x;
  const {{index_type}} dense_idx_elem = dense_idx * N_ELEMENTS_PER_THREAD;
  if (dense_idx_elem >= n_elements) {
    return;
  }

  // then compute the jagged_idx from the dense_idx_elem
  {{index_type}} jagged_idx;
  {
    // dense_coord is along consecutive dense dimensions
    // jagged_coord is along the total_length of the jagged Tensor
    {{index_type}} dense_coord = dense_idx_elem / ({{strides[0]}});
    {{index_type}} running_idx = dense_idx_elem % ({{strides[0]}});
    {{offsets_type}} jagged_coord = 0, prev_offset, next_offset;

{% for i in range(num_offsets) %}
    prev_offset = offsets.data[{{i}}][jagged_coord + dense_coord];
    next_offset = offsets.data[{{i}}][jagged_coord + dense_coord + 1];
    dense_coord = running_idx / ({{strides[i+1]}});
    running_idx = running_idx % ({{strides[i+1]}});
    if (dense_coord >= next_offset - prev_offset) {
        // this element of the dense volume is
        // out of bounds of the jagged Tensor
        return;
    }
    jagged_coord = prev_offset;

{% endfor %}
    jagged_coord += dense_coord;
    jagged_idx = (jagged_coord * ({{strides[num_offsets]}}) + running_idx) / N_ELEMENTS_PER_THREAD;
  }
    """
)

KERNEL_COMPUTE_JAGGED_IDX_THEN_DENSE_IDX_TEMPLATE = jinja2.Template(
    """
  // first compute the jagged_idx from the blockIdx and threadIdx
  const {{index_type}} jagged_idx = blockIdx.x * FUSED_ELE_THREAD_SIZE + threadIdx.x;
  const {{index_type}} jagged_idx_elem = jagged_idx * N_ELEMENTS_PER_THREAD;
  if (jagged_idx_elem >= n_elements) {
    return;
  }

  // then compute the dense_idx from the jagged_idx_elem
  {{index_type}} dense_idx = jagged_idx_elem % ({{strides[num_offsets]}});
  {
    {{offsets_type}} left, right, mid, tmp_value, offset_idx, offset_value;
    {{index_type}} running_idx = jagged_idx_elem / ({{strides[num_offsets]}});

    // binary search to determine the dense coord along the current jagged dimension
    // the goal is to find the index of the maximum offset value in offsets.data[{{i}}]
    // which is <= the running_idx. the (running_idx - offset_value) will then indicate
    // the dense cooord along the current jagged dimension.
{% for i in range(num_offsets - 1, -1, -1) %}
    left = 0;
    right = offsets.lengths[{{i}}] - 1;
    while (left <= right) {
        mid = (left + right) >> 1;
        tmp_value = offsets.data[{{i}}][mid];
        if (tmp_value <= running_idx) {
            offset_idx = mid;
            offset_value = tmp_value;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    dense_idx += (running_idx - offset_value) * ({{strides[i+1]}});
    running_idx = offset_idx;

{% endfor %}
    dense_idx = (dense_idx + running_idx * ({{strides[0]}})) / N_ELEMENTS_PER_THREAD;
  }
    """
)

KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void {{func_name}}(
    {{read_t}}* y,
    const {{read_t}}* x,
    {{dynamic_dims}}
    {{offsets}}
    {{index_type}} n_elements
) {
  {{compute_idx}}

  y[jagged_idx] = x[dense_idx];
}
    """
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{head}}

#include <iostream>
#include "jagged.h"

namespace {

{{constant}}

{{kernel_function}}

}  // namespace

void {{func_name}}(
    void* y,
    const void* x,
{% for idx in range(num_offsets) %}
    {{index_type}} offsets_length_{{idx}},
    const void* offsets_data_{{idx}},
{% endfor %}
    {{dynamic_dims_decl}}
    {{prefix}}Stream_t stream
) {
    {{index_type}} n_elements = {{calculate_n}};
    if (n_elements == 0) {
      return;
    }

    // we define local offsets here, because the resulting jagged Tensor's offsets
    // haven't been initialized by make_jagged yet, which is invoked after this op
    {{offsets_struct_type}} local_offsets;
{% for idx in range(num_offsets) %}
    local_offsets.lengths[{{idx}}] = offsets_length_{{idx}};
    local_offsets.data[{{idx}}] = reinterpret_cast<const {{offsets_type}}*>(offsets_data_{{idx}});
{% endfor %}

    int block_size = static_cast<int>(std::ceil(static_cast<double>(n_elements) / N_ELEMENTS_PER_THREAD / FUSED_ELE_THREAD_SIZE));
    {{func_name}}<<<block_size, FUSED_ELE_THREAD_SIZE, 0, stream>>>(
        reinterpret_cast<{{read_t}}*>(y),
        reinterpret_cast<const {{read_t}}*>(x),
        {{dynamic_dims_call}}
        local_offsets,
        n_elements
    );
}
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void* y,
    const void* x,
{% for idx in range(num_offsets) %}
    {{index_type}},
    const void*,
{% endfor %}
    {{dynamic_dims}}
    {{prefix}}Stream_t stream
);
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{y}},
{{indent}}    {{x}},
{% for idx in range(num_offsets) %}
{{indent}}    {{offsets_first_dim_names[idx]}},
{{indent}}    {{offsets_data_names[idx]}},
{% endfor %}
{{indent}}    {{dynamic_dims}}
{{indent}}    {{stream}}
{{indent}});
    """
)


def _get_strides(shape: List[IntVar]) -> List[str]:
    """
    Generate the stride expressions for each of the dimensions
    of the shape. A stride expression here means the
    product of all dimensions following the given dimension.
    The order of the stride expressions in the returned list
    is the same as of the dimensions of the shape.
    """
    strides = []
    for dim in reversed(shape[1:]):
        str_dim = str(dim.value()) if isinstance(dim, IntImm) else dim._attrs["name"]
        if strides:
            strides.append(f"{strides[-1]} * {str_dim}")
        else:
            strides.append(str_dim)
    strides.reverse()
    return strides


def _get_dynamic_dims(x: Tensor, y: Tensor) -> List[IntVar]:
    res = {}
    for dim in list(x.shape()) + list(y.shape()):
        if not isinstance(dim, IntImm):
            res[dim._attrs["name"]] = dim

    return list(res.values())


def _gen_dynamic_dim_str(
    index_type: str,
    dynamic_dims: List[IntVar],
    has_type: bool,
) -> str:
    type_str = index_type + " " if has_type else ""
    res = ", ".join([type_str + dim._attrs["name"] for dim in dynamic_dims])
    if res:
        res += ", "

    return res


def _gen_offsets_str(
    jagged_int_var: JaggedIntVar,
    has_type: bool,
    const_ref: bool,
    name: Optional[str] = None,
) -> str:
    offsets_var_name = jagged_int_var.offsets_var_name()
    offsets_struct_type = jagged_int_var.offsets_struct_type()

    ref_prefix = "const " if const_ref else ""
    ref_suffix = "&" if const_ref else ""
    arg_type = f"{ref_prefix}{offsets_struct_type}{ref_suffix} " if has_type else ""
    arg_name = name if name is not None else offsets_var_name
    offsets = f"{arg_type}{arg_name}, "

    return offsets


def _gen_int_var_product_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )

    return " * ".join(res) if res else "1"


def _detect_read_type(
    inner_size: int,
    dtype: str,
) -> str:
    if dtype in ("bfloat16", "half"):
        if inner_size % 8 == 0:
            return "uint4"
        elif inner_size % 4 == 0:
            return "uint2"
        elif inner_size % 2 == 0:
            return "uint"
    elif dtype == "float":
        if inner_size % 4 == 0:
            return "uint4"
        elif inner_size % 2 == 0:
            return "uint2"

    return dtype


def _gen_compute_idx_str(
    input_shape: List[IntVar],
    output_shape: List[IntVar],
    index_type: str,
    jagged_int_var: JaggedIntVar,
) -> str:
    use_jagged_space_indexing = Target.current()._kwargs.get(
        "use_jagged_space_indexing", False
    )
    compute_idx_template = (
        KERNEL_COMPUTE_JAGGED_IDX_THEN_DENSE_IDX_TEMPLATE
        if use_jagged_space_indexing
        else KERNEL_COMPUTE_DENSE_IDX_THEN_JAGGED_IDX_TEMPLATE
    )

    return compute_idx_template.render(
        index_type=index_type,
        num_offsets=len(jagged_int_var.jagged_dims()),
        strides=_get_strides(input_shape),
        offsets_type=jagged_int_var.offsets_type(),
    )


def _gen_calculate_n(
    input_shape: List[IntVar],
    output_shape: List[IntVar],
) -> str:
    use_jagged_space_indexing = Target.current()._kwargs.get(
        "use_jagged_space_indexing", False
    )
    # we use jagged output's volume in case of the jagged space indexing
    # and dense input's volume in case of the dense space indexing
    index_space = output_shape if use_jagged_space_indexing else input_shape

    return _gen_int_var_product_str(index_space)


def _gen_kernel_function(
    func_attrs: Dict[str, Any],
    index_type: str,
    data_type: str,
    read_type: str,
) -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = func_attrs["jagged_int_var"]
    backend_spec = CUDASpec()

    return KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        compute_idx=_gen_compute_idx_str(
            input_shape=x.shape(),
            output_shape=y.shape(),
            index_type=index_type,
            jagged_int_var=jagged_int_var,
        ),
        read_t=read_type,
        dynamic_dims=_gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=_get_dynamic_dims(x, y),
            has_type=True,
        ),
        offsets=_gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=True,
            # the offsets are passed
            # by value to the kernel
            const_ref=False,
            name="offsets",
        ),
    )


@registry.reg("cuda.dense_to_jagged.gen_function")
def dense_to_jagged_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates dense_to_jagged function definition."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = func_attrs["jagged_int_var"]
    backend_spec = CUDASpec()

    dtype = x.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)
    read_inner_size = shape_utils.get_num_rightmost_static_elements(y.shape())
    read_type = _detect_read_type(read_inner_size, data_type)

    kernel_function = _gen_kernel_function(
        func_attrs,
        backend_spec.index_type,
        data_type,
        read_type,
    )

    constant = CONSTANT_TEMPLATE.render(
        read_t=read_type,
        data_t=data_type,
    )

    func_name = func_attrs["name"]
    dynamic_dims = _get_dynamic_dims(x, y)
    offsets_struct_type = jagged_int_var.offsets_struct_type()
    total_length = jagged_int_var.total_length()

    if total_length._attrs.get("isolated", False):
        raise ValueError(
            f"The {total_length._attrs['name']} (total_length) dimension "
            f"of the jagged Tensor output of {func_name} must be present in "
            "one of the input shapes, but it isn't."
        )

    return FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(),
        offsets_struct_type=offsets_struct_type,
        offsets_type=jagged_int_var.offsets_type(),
        num_offsets=len(jagged_int_var.jagged_dims()),
        constant=constant,
        kernel_function=kernel_function,
        func_name=func_name,
        calculate_n=_gen_calculate_n(
            input_shape=x.shape(),
            output_shape=y.shape(),
        ),
        dynamic_dims_decl=_gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=dynamic_dims,
            has_type=True,
        ),
        dynamic_dims_call=_gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=dynamic_dims,
            has_type=False,
        ),
        read_t=read_type,
    )


@registry.reg("cuda.dense_to_jagged.func_decl")
def dense_to_jagged_gen_function_decl(func_attrs) -> str:
    """Generate dense_to_jagged function declaration."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = func_attrs["jagged_int_var"]
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    return FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        num_offsets=len(jagged_int_var.jagged_dims()),
        dynamic_dims=_gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=_get_dynamic_dims(x, y),
            has_type=True,
        ),
    )


@registry.reg("cuda.dense_to_jagged.func_call")
def dense_to_jagged_gen_function_call(
    func_attrs,
    indent: str,
) -> str:
    """Generate dense_to_jagged function call."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = func_attrs["jagged_int_var"]
    backend_spec = CUDASpec()

    offsets_list = func_attrs["inputs"][1:]
    offsets_first_dim_names = [
        offsets._attrs["shape"][0]._attrs["name"] for offsets in offsets_list
    ]
    offsets_data_names = [offsets._attrs["name"] for offsets in offsets_list]

    return FUNC_CALL_TEMPLATE.render(
        stream=backend_spec.stream,
        func_name=func_attrs["name"],
        num_offsets=len(jagged_int_var.jagged_dims()),
        offsets_first_dim_names=offsets_first_dim_names,
        offsets_data_names=offsets_data_names,
        y=y._attrs["name"],
        x=x._attrs["name"],
        dynamic_dims=_gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=_get_dynamic_dims(x, y),
            has_type=False,
        ),
        indent=indent,
    )
