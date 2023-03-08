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
Define jagged_to_dense codegen and CUDA kernel
"""
from typing import Any, Dict, List, Optional

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.compiler.base import IntImm, IntVar, Tensor
from aitemplate.utils import shape_utils


CONSTANT_TEMPLATE = jinja2.Template(
    """
#define FUSED_ELE_THREAD_SIZE 256

const int N_ELEMENTS_PER_THREAD = sizeof({{read_t}}) / sizeof({{data_t}});
    """
)

KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void {{func_name}}({{read_t}}* y, const {{read_t}}* x, {{dynamic_dims}} {{offsets}} {{index_type}} n_elements) {
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
        {{read_t}} padded_vector;
        {{data_t}}* cursor = reinterpret_cast<{{data_t}}*>(&padded_vector);

        #pragma unroll
        for (int i = 0; i < N_ELEMENTS_PER_THREAD; i++) {
            cursor[i] = {{data_t}}({{padding_value}});
        }

        y[dense_idx] = padded_vector;
        return;
    }
    jagged_coord = prev_offset;

{% endfor %}
    jagged_coord += dense_coord;
    jagged_idx = (jagged_coord * ({{strides[num_offsets]}}) + running_idx) / N_ELEMENTS_PER_THREAD;
  }
  y[dense_idx] = x[jagged_idx];
}
    """
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{head}}

#include "jagged.h"

namespace {

{{constant}}

{{kernel_function}}

}  // namespace

void invoke_{{func_name}}(void* y, const void* x, {{dynamic_dims_decl}} {{offsets_decl}} {{index_type}} n_elements, {{prefix}}Stream_t stream) {
    if (n_elements == 0) {
      return;
    }
    int block_size = static_cast<int>(std::ceil(static_cast<double>(n_elements) / N_ELEMENTS_PER_THREAD / FUSED_ELE_THREAD_SIZE));
    {{func_name}}<<<block_size, FUSED_ELE_THREAD_SIZE, 0, stream>>>(
        reinterpret_cast<{{read_t}}*>(y),
        reinterpret_cast<const {{read_t}}*>(x),
        {{dynamic_dims_call}}
        {{offsets_call}}
        n_elements
    );
}
    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void invoke_{{func_name}}(void* y, const void* x, {{dynamic_dims}} {{offsets}} {{index_type}} n_elements, {{prefix}}Stream_t stream);
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}{{index_type}} {{func_name}}_n_elements = {{calculate_n}};
    {{indent}}invoke_{{func_name}}({{y}}, {{x}}, {{dynamic_dims}} {{offsets}} {{func_name}}_n_elements, {{stream}});
{{indent}}}
    """
)


def _get_output_volume_strides(
    output_volume: List[IntVar],
) -> List[str]:
    """
    Generate the stride expressions for each of the dimensions
    of the y volume. A stride expression here means the
    product of all dimensions following the given dimension.
    The order of the stride expressions in the returned list
    is the same as of the dimensions of the y volume.
    """
    strides = []
    for dim in reversed(output_volume[1:]):
        str_dim = str(dim.value()) if isinstance(dim, IntImm) else dim._attrs["name"]
        if strides:
            strides.append(f"{strides[-1]} * {str_dim}")
        else:
            strides.append(str_dim)
    strides.reverse()
    return strides


def _get_dynamic_dims(y: Tensor) -> List[IntVar]:
    res = {}

    for dim in y.shape():
        if not isinstance(dim, IntImm):
            res[dim._attrs["name"]] = dim
    return list(res.values())


def _gen_dynamic_dim_str(
    index_type: str, dynamic_dims: List[IntVar], has_type: bool
) -> str:
    type_str = index_type + " " if has_type else ""
    res = ", ".join([type_str + dim._attrs["name"] for dim in dynamic_dims])
    if res:
        res += ", "
    return res


def _gen_offsets_str(
    x: Tensor,
    has_type: bool,
    const_ref: bool,
    name: Optional[str] = None,
) -> str:
    jagged_int_var = x._attrs["shape"][0]
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


def _detect_read_type(inner_size: int, dtype: str) -> str:
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


def _gen_kernel_function(
    func_attrs: Dict[str, Any],
    index_type: str,
    data_type: str,
    read_type: str,
) -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    padding_value = func_attrs["padding_value"]
    jagged_int_var = x.shape()[0]
    num_offsets = len(jagged_int_var.jagged_dims())
    backend_spec = CUDASpec()

    dynamic_dims = _get_dynamic_dims(y)

    kernel_func = KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        num_offsets=num_offsets,
        strides=_get_output_volume_strides(
            y.shape(),
        ),
        offsets_type=jagged_int_var.offsets_type(),
        data_t=data_type,
        read_t=read_type,
        padding_value=padding_value,
        dynamic_dims=_gen_dynamic_dim_str(
            backend_spec.index_type,
            dynamic_dims,
            has_type=True,
        ),
        offsets=_gen_offsets_str(
            x,
            has_type=True,
            # the offsets are passed
            # by value to the kernel
            const_ref=False,
            name="offsets",
        ),
    )
    return kernel_func


@registry.reg("cuda.jagged_to_dense.gen_function")
def jagged_to_dense_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates jagged_to_dense function definition."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    backend_spec = CUDASpec()

    dtype = x.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)
    read_inner_size = shape_utils.get_num_rightmost_static_elements(x.shape())
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

    dynamic_dims = _get_dynamic_dims(y)

    function = FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(),
        constant=constant,
        kernel_function=kernel_function,
        func_name=func_attrs["name"],
        dynamic_dims_decl=_gen_dynamic_dim_str(
            backend_spec.index_type,
            dynamic_dims,
            has_type=True,
        ),
        dynamic_dims_call=_gen_dynamic_dim_str(
            backend_spec.index_type,
            dynamic_dims,
            has_type=False,
        ),
        offsets_decl=_gen_offsets_str(
            x,
            has_type=True,
            # the offsets are passed
            # by const reference to the function
            const_ref=True,
            name="offsets",
        ),
        offsets_call=_gen_offsets_str(
            x,
            has_type=False,
            const_ref=False,
            name="offsets",
        ),
        read_t=read_type,
    )
    return function


@registry.reg("cuda.jagged_to_dense.func_decl")
def jagged_to_dense_gen_function_decl(func_attrs) -> str:
    """Generate jagged_to_dense function declaration."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    dynamic_dims = _get_dynamic_dims(y)

    return FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        dynamic_dims=_gen_dynamic_dim_str(
            backend_spec.index_type,
            dynamic_dims,
            has_type=True,
        ),
        offsets=_gen_offsets_str(
            x,
            has_type=True,
            const_ref=True,
            name="offsets",
        ),
    )


@registry.reg("cuda.jagged_to_dense.func_call")
def jagged_to_dense_gen_function_call(
    func_attrs,
    indent: str,
) -> str:
    """Generate jagged_to_dense function call."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    backend_spec = CUDASpec()
    dynamic_dims = _get_dynamic_dims(y)

    return FUNC_CALL_TEMPLATE.render(
        stream=backend_spec.stream,
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        calculate_n=_gen_int_var_product_str(
            y.shape(),
        ),
        y=y._attrs["name"],
        x=x._attrs["name"],
        dynamic_dims=_gen_dynamic_dim_str(
            backend_spec.index_type,
            dynamic_dims,
            has_type=False,
        ),
        offsets=_gen_offsets_str(
            x,
            has_type=False,
            const_ref=False,
        ),
        indent=indent,
    )
