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
The back-end bindings of the padded_dense_to_jagged op.
"""
from typing import Any, Dict, List

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.elementwise_common import (
    CONSTANT_TEMPLATE,
    gen_dynamic_dim_str,
    gen_int_var_product_str,
    gen_offsets_str,
    get_dynamic_dims,
    get_stride_expressions,
    KERNEL_COMPUTE_DENSE_IDX_THEN_JAGGED_IDX_TEMPLATE,
    KERNEL_COMPUTE_JAGGED_IDX_THEN_DENSE_IDX_TEMPLATE,
)
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntVar, JaggedIntVar
from aitemplate.utils import shape_utils


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

void invoke_{{func_name}}(
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
void invoke_{{func_name}}(
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
{{indent}}invoke_{{func_name}}(
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
        strides=get_stride_expressions(input_shape),
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

    return gen_int_var_product_str(index_space)


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
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(x.shape(), y.shape()),
            has_type=True,
        ),
        offsets=gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=True,
            # the offsets are passed
            # by value to the kernel
            const_ref=False,
            name="offsets",
        ),
    )


@registry.reg("cuda.padded_dense_to_jagged.gen_function")
def padded_dense_to_jagged_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates padded_dense_to_jagged function definition."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = func_attrs["jagged_int_var"]
    backend_spec = CUDASpec()

    dtype = x.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)

    # inner size of the output jagged Tensor: can't use the input dense Tensor
    # shape here, as some the dimensions in it may overlap with the jagged
    # dimensions of the output jagged Tensor
    inner_size = shape_utils.get_num_rightmost_static_elements(y.shape())
    read_type = backend_spec.get_elementwise_read_backend_type(inner_size, dtype)

    kernel_function = _gen_kernel_function(
        func_attrs=func_attrs,
        index_type=backend_spec.index_type,
        data_type=data_type,
        read_type=read_type,
    )

    constant = CONSTANT_TEMPLATE.render(
        read_t=read_type,
        data_t=data_type,
        op_t=data_type,
    )

    func_name = func_attrs["name"]
    dynamic_dims = get_dynamic_dims(x.shape(), y.shape())
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
        dynamic_dims_decl=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=dynamic_dims,
            has_type=True,
        ),
        dynamic_dims_call=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=dynamic_dims,
            has_type=False,
        ),
        read_t=read_type,
    )


@registry.reg("cuda.padded_dense_to_jagged.func_decl")
def padded_dense_to_jagged_gen_function_decl(func_attrs) -> str:
    """Generate padded_dense_to_jagged function declaration."""

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
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(x.shape(), y.shape()),
            has_type=True,
        ),
    )


@registry.reg("cuda.padded_dense_to_jagged.func_call")
def padded_dense_to_jagged_gen_function_call(
    func_attrs,
    indent: str,
) -> str:
    """Generate padded_dense_to_jagged function call."""

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
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(x.shape(), y.shape()),
            has_type=False,
        ),
        indent=indent,
    )
