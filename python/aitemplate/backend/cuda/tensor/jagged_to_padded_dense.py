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
The back-end bindings of the jagged_to_padded_dense op.
"""
from typing import Any, Dict

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
)
from aitemplate.utils import shape_utils


KERNEL_PADDING_TEMPLATE = jinja2.Template(
    """
        {{read_t}} padded_vector;
        {{data_t}}* cursor = reinterpret_cast<{{data_t}}*>(&padded_vector);

        #pragma unroll
        for (int i = 0; i < N_ELEMENTS_PER_THREAD; i++) {
            cursor[i] = {{data_t}}({{padding_value}});
        }

        y[dense_idx] = padded_vector;
    """
)


KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void {{func_name}}({{read_t}}* y, const {{read_t}}* x, {{dynamic_dims}} {{offsets}} {{index_type}} n_elements) {
  {{compute_idx}}

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

    padding_str = KERNEL_PADDING_TEMPLATE.render(
        data_t=data_type,
        read_t=read_type,
        padding_value=padding_value,
    )

    compute_idx_str = KERNEL_COMPUTE_DENSE_IDX_THEN_JAGGED_IDX_TEMPLATE.render(
        index_type=index_type,
        num_offsets=num_offsets,
        strides=get_stride_expressions(y.shape()),
        offsets_type=jagged_int_var.offsets_type(),
        out_of_bounds_action=padding_str,
    )

    return KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=index_type,
        read_t=read_type,
        compute_idx=compute_idx_str,
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(y.shape()),
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


@registry.reg("cuda.jagged_to_padded_dense.gen_function")
def jagged_to_padded_dense_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates jagged_to_padded_dense function definition."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = x.shape()[0]
    backend_spec = CUDASpec()

    dtype = x.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)

    # inner size of the input jagged Tensor: can't use the output dense Tensor
    # shape here, as some the dimensions in it may overlap with the jagged
    # dimensions of the input jagged Tensor
    inner_size = shape_utils.get_num_rightmost_static_elements(x.shape())
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

    dynamic_dims = get_dynamic_dims(y.shape())

    return FUNC_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        head=backend_spec.header_src_template.render(),
        constant=constant,
        kernel_function=kernel_function,
        func_name=func_attrs["name"],
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
        offsets_decl=gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=True,
            # the offsets are passed
            # by const reference to the function
            const_ref=True,
            name="offsets",
        ),
        offsets_call=gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=False,
            const_ref=False,
            name="offsets",
        ),
        read_t=read_type,
    )


@registry.reg("cuda.jagged_to_padded_dense.func_decl")
def jagged_to_padded_dense_gen_function_decl(func_attrs) -> str:
    """Generate jagged_to_padded_dense function declaration."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = x.shape()[0]
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    return FUNC_DECL_TEMPLATE.render(
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        func_name=func_name,
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(y.shape()),
            has_type=True,
        ),
        offsets=gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=True,
            const_ref=True,
            name="offsets",
        ),
    )


@registry.reg("cuda.jagged_to_padded_dense.func_call")
def jagged_to_padded_dense_gen_function_call(
    func_attrs,
    indent: str,
) -> str:
    """Generate jagged_to_padded_dense function call."""

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    jagged_int_var = x.shape()[0]
    backend_spec = CUDASpec()

    return FUNC_CALL_TEMPLATE.render(
        stream=backend_spec.stream,
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        calculate_n=gen_int_var_product_str(y.shape()),
        y=y._attrs["name"],
        x=x._attrs["name"],
        dynamic_dims=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(y.shape()),
            has_type=False,
        ),
        offsets=gen_offsets_str(
            jagged_int_var=jagged_int_var,
            has_type=False,
            const_ref=False,
        ),
        indent=indent,
    )
