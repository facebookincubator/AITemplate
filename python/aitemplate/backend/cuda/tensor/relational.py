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
#define N_READS_PER_THREAD sizeof({{output_read_t}}) / sizeof(bool)
    """
)


FUNC_DECL = jinja2.Template(
    """

void invoke_{{func_name}}(
    void*,  /* output */
    const void*,  /* left operand */
{% if not right_operand.is_a_const_num() %}
    const void*,   /* right operand */
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
    {{indent}}    {{left_operand_name}},
{% if not right_operand.is_a_const_num() %}
    {{indent}}    {{right_operand_name}},
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

__global__ void relational(
    {{output_read_t}}* output,
    const {{input_read_t}}* left_operand,
{% if not right_operand.is_a_const_num() %}
    const {{input_read_t}}* right_operand,
{% endif %}
    {{index_type}} num_elements) {

    const {{index_type}} idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx * N_READS_PER_THREAD >= num_elements) {
        return;
    }

    {{input_read_t}} tmp_left = left_operand[idx];
    {{data_type}}* tmp_left_ptr = reinterpret_cast<{{data_type}}*>(&tmp_left);

    {{output_read_t}} tmp_output;
    bool* tmp_output_ptr = reinterpret_cast<bool*>(&tmp_output);

{% if not right_operand.is_a_const_num() %}
    {{input_read_t}} tmp_right = right_operand[idx];
    {{data_type}}* tmp_right_ptr = reinterpret_cast<{{data_type}}*>(&tmp_right);
{% endif %}

  #pragma unroll
    for (int i=0; i < N_READS_PER_THREAD; i++) {

{% if not right_operand.is_a_const_num() %}
        tmp_output_ptr[i] = (tmp_left_ptr[i] {{operator}} tmp_right_ptr[i]);
{% else %}
        tmp_output_ptr[i] = (tmp_left_ptr[i] {{operator}} ({{data_type}})({{right_operand_value}}));
{% endif %}
  }
    output[idx] = tmp_output;
}

} // namespace

void invoke_{{func_name}}(
    void* output,
    const void* input_1,
{% if not right_operand.is_a_const_num() %}
    const void* input_2,
{% endif %}
    {{index_type}} num_elements,
    {{prefix}}Stream_t stream) {

  int grid_size = static_cast<int>(
      std::ceil(static_cast<double>(num_elements) / N_THREADS_PER_BLOCK / N_READS_PER_THREAD));

  relational<<<grid_size, N_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<{{output_read_t}}*>(output),
      reinterpret_cast<const {{input_read_t}}*>(input_1),
{% if not right_operand.is_a_const_num() %}
      reinterpret_cast<const {{input_read_t}}*>(input_2),
{% endif %}
      num_elements);
}
    """
)


@registry.reg("cuda.relational.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    inputs = func_attrs["inputs"]
    backend_spec = CUDASpec()

    input_dtype = inputs[0].dtype()
    input_read_t = backend_spec.get_elementwise_read_backend_type(
        shape_utils.get_num_rightmost_static_elements(inputs[0].shape()), input_dtype
    )
    input_data_t = backend_spec.dtype_to_backend_type(input_dtype)
    read_vector_length = (
        backend_spec.sizeof_types[input_read_t]
        / backend_spec.sizeof_types[input_data_t]
    )
    # output data type is bool, which is 1 byte
    output_read_t = {
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
        constant=CONSTANT_TEMPLATE.render(output_read_t=output_read_t),
        func_name=func_attrs["name"],
        data_type=input_data_t,
        index_type=backend_spec.index_type,
        operator=func_attrs["func"].value,
        prefix=backend_spec.prefix,
        right_operand=func_attrs["args"][1],
        right_operand_value=str(func_attrs["args"][1]._attrs["value"]),
        output_read_t=output_read_t,
        input_read_t=input_read_t,
    )


@registry.reg("cuda.relational.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    backend_spec = CUDASpec()
    return FUNC_DECL.render(
        func_name=func_attrs["name"],
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        right_operand=func_attrs["args"][1],
    )


@registry.reg("cuda.relational.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    y = func_attrs["outputs"][0]
    backend_spec = CUDASpec()
    args = func_attrs["args"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        left_operand_name=args[0]._attrs["name"],
        right_operand_name=args[1]._attrs["name"],
        right_operand=args[1],
        calculate_n=gen_int_var_product_str(y._attrs["shape"]),
        indent=indent,
        index_type=backend_spec.index_type,
    )
