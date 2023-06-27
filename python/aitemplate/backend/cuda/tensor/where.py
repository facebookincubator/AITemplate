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

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.elementwise_common import gen_int_var_product_str
from aitemplate.utils import shape_utils


CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
"""


CONSTANT_TEMPLATE = jinja2.Template(
    """
#define N_THREADS_PER_BLOCK 256
#define N_READS_PER_THREAD sizeof({{condition_read_t}}) / sizeof(bool)
    """
)


FUNC_DECL = jinja2.Template(
    """

void invoke_{{func_name}}(
    void*,  /* output */
    const void*,  /* condition */
{% if not input_tensor_is_a_const_num %}
    const void*,  /* input tensor */
{% endif %}
{% if not other_tensor_is_a_const_num %}
    const void*,   /* other tensor */
{% endif %}
    {{index_type}}, /* number of elements */
    {{prefix}}Stream_t  /* stream */
);
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}{{index_type}} n_elements = {{calculate_n}};
    {{indent}} invoke_{{func_name}}(
    {{indent}}    {{output}},
    {{indent}}    {{condition}},
{% if not input_tensor_is_a_const_num %}
    {{indent}}    {{input_tensor}},
{% endif %}
{% if not other_tensor_is_a_const_num %}
    {{indent}}    {{other_tensor}},
{% endif %}
    {{indent}}    n_elements,
    {{indent}}    stream
    {{indent}});
{{indent}}}
    """
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{constant}}


__global__ void where(
    {{read_t}}* output,
    const {{condition_read_t}}* condition,
{% if not input_tensor_is_a_const_num %}
    const {{read_t}}* input_tesnor,
{% endif %}
{% if not other_tensor_is_a_const_num %}
    const {{read_t}}* other_tensor,
{% endif %}
    {{index_type}} num_elements) {
        const {{index_type}} idx = (blockIdx.x * blockDim.x + threadIdx.x);
        if (idx * N_READS_PER_THREAD >= num_elements) {
            return;
        }

        {{read_t}} tmp_output;
        {{data_t}}* tmp_output_ptr = reinterpret_cast<{{data_t}}*>(&tmp_output);

        {{condition_read_t}} tmp_condition = condition[idx];
        bool* tmp_condition_ptr = reinterpret_cast<bool*>(&tmp_condition);

{% if not input_tensor_is_a_const_num %}
        {{read_t}} tmp_input_tensor = input_tesnor[idx];
        {{data_t}}* tmp_input_tensor_ptr = reinterpret_cast<{{data_t}}*>(&tmp_input_tensor);
{% endif %}

{% if not other_tensor_is_a_const_num %}
        {{read_t}} tmp_other_tensor = other_tensor[idx];
        {{data_t}}* tmp_other_tensor_ptr = reinterpret_cast<{{data_t}}*>(&tmp_other_tensor);
{% endif %}

#pragma unroll
        for (int i=0; i < N_READS_PER_THREAD; i++) {
            tmp_output_ptr[i] = ({{data_t}})(tmp_condition_ptr[i]) * ({{data_t}})({{ input_tensor_val if input_tensor_is_a_const_num else "tmp_input_tensor_ptr[i]" }}) + ({{data_t}})(1 - tmp_condition_ptr[i]) * ({{data_t}})({{ other_tensor_val if other_tensor_is_a_const_num else "tmp_other_tensor_ptr[i]" }});
        }
        output[idx] = tmp_output;

    }

} // namespace

void invoke_{{func_name}}(
    void* output,
    const void* condition,
{% if not input_tensor_is_a_const_num %}
    const void* input_tesnor,
{% endif %}
{% if not other_tensor_is_a_const_num %}
    const void* other_tensor,
{% endif %}
    {{index_type}} num_elements,
    {{prefix}}Stream_t stream) {

  int grid_size = static_cast<int>(
      std::ceil(static_cast<double>(num_elements) / N_THREADS_PER_BLOCK / N_READS_PER_THREAD));

  where<<<grid_size, N_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<{{read_t}}*>(output),
      reinterpret_cast<const {{condition_read_t}}*>(condition),
{% if not input_tensor_is_a_const_num %}
      reinterpret_cast<const {{read_t}}*>(input_tesnor),
{% endif %}
{% if not other_tensor_is_a_const_num %}
      reinterpret_cast<const {{read_t}}*>(other_tensor),
{% endif %}
      num_elements);
}
    """
)


@registry.reg("cuda.where.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    condition, input_tensor, other_tensor = func_attrs["args"]
    output = func_attrs["outputs"][0]
    dtype = output.dtype()
    backend_spec = CUDASpec()
    read_t = backend_spec.get_elementwise_read_backend_type(
        shape_utils.get_num_rightmost_static_elements(output.shape()), dtype
    )
    data_t = backend_spec.dtype_to_backend_type(dtype)
    read_vector_length = (
        backend_spec.sizeof_types[read_t] / backend_spec.sizeof_types[data_t]
    )
    # condition data type is bool, which is 1 byte
    condition_read_t = {
        1: "bool",
        2: "half",
        4: "float",
        8: "int2",
        16: "int4",
    }[read_vector_length]

    return FUNC_TEMPLATE.render(
        header_files=backend_spec.header_src_template.render(
            extra_header=CUDA_HEADER_FILES
        ),
        constant=CONSTANT_TEMPLATE.render(condition_read_t=condition_read_t),
        func_name=func_attrs["name"],
        data_t=data_t,
        read_t=read_t,
        condition_read_t=condition_read_t,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        input_tensor_is_a_const_num=input_tensor.is_a_const_num(),
        other_tensor_is_a_const_num=other_tensor.is_a_const_num(),
        input_tensor_val=str(input_tensor._attrs["value"]),
        other_tensor_val=str(other_tensor._attrs["value"]),
    )


@registry.reg("cuda.where.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _, input_tensor, other_tensor = func_attrs["args"]
    backend_spec = CUDASpec()
    return FUNC_DECL.render(
        func_name=func_attrs["name"],
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        input_tensor_is_a_const_num=input_tensor.is_a_const_num(),
        other_tensor_is_a_const_num=other_tensor.is_a_const_num(),
    )


@registry.reg("cuda.where.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    condition, input_tensor, other_tensor = func_attrs["args"]
    output = func_attrs["outputs"][0]
    backend_spec = CUDASpec()
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output._attrs["name"],
        condition=condition._attrs["name"],
        input_tensor=input_tensor._attrs["name"],
        other_tensor=other_tensor._attrs["name"],
        calculate_n=gen_int_var_product_str(condition.shape()),
        indent=indent,
        index_type=backend_spec.index_type,
        input_tensor_is_a_const_num=input_tensor.is_a_const_num(),
        other_tensor_is_a_const_num=other_tensor.is_a_const_num(),
    )
