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

CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
"""

CONSTANT_TEMPLATE = jinja2.Template(
    """
#define N_THREADS_PER_BLOCK 256

    """
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void invoke_{{func_name}}(
    void* y,
    const void* x,
    {{index_type}} n_elements,
    {{prefix}}Stream_t stream);
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}const {{index_type}} {{func_name}}_n_elements = {{calculate_n}};
    {{indent}}invoke_{{func_name}}({{output}}, {{input}},  {{func_name}}_n_elements, stream);
{{indent}}}
    """
)


FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{constant}}

__global__  void cast_op(
    {{output_type}}* output,
    const {{input_type}}* input,
    {{index_type}} n_elements
) {
    const {{index_type}} idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_elements) {
        return;
    }
    output[idx] = {{cast_func_call}}
  }

}  // namespace

void invoke_{{func_name}}(void* output, const void* input,
    {{index_type}} n_elements, {{prefix}}Stream_t stream) {
    if (n_elements == 0) {
      return;
    }
    int grid_size = static_cast<int>(std::ceil(static_cast<double>(n_elements) / N_THREADS_PER_BLOCK));
    cast_op<<<grid_size, N_THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<{{output_type}}*>(output),
        reinterpret_cast<const {{input_type}}*>(input),
        n_elements
    );
}
    """
)

CAST_FUNCS = {
    "half": {
        "bfloat16": "__float2bfloat16_rn(__half2float(input[idx]));",
        "float": "__half2float(input[idx]);",
    },
    "bfloat16": {
        "half": "__float2half_rn(__bfloat162float(input[idx]));",
        "float": "__bfloat162float(input[idx]);",
    },
    "float": {
        "bfloat16": "__float2bfloat16_rn(input[idx]);",
        "half": "__float2half_rn(input[idx]);",
    },
}


@registry.reg("cuda.cast.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    input_ = func_attrs["inputs"][0]
    output = func_attrs["outputs"][0]
    backend_spec = CUDASpec()
    output_dtype = output.dtype()
    output_type = backend_spec.dtype_to_backend_type(output_dtype)
    input_type = backend_spec.dtype_to_backend_type(input_.dtype())
    cast_func_call = CAST_FUNCS[input_type][output_type]

    return FUNC_TEMPLATE.render(
        header_files=backend_spec.header_src_template.render(
            extra_header=CUDA_HEADER_FILES
        ),
        constant=CONSTANT_TEMPLATE.render(),
        func_name=func_attrs["name"],
        input_type=input_type,
        output_type=output_type,
        index_type=backend_spec.index_type,
        cast_func_call=cast_func_call,
        prefix=backend_spec.prefix,
    )


@registry.reg("cuda.cast.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    backend_spec = CUDASpec()
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
    )


@registry.reg("cuda.cast.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    backend_spec = CUDASpec()
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=func_attrs["outputs"][0]._attrs["name"],
        input=func_attrs["inputs"][0]._attrs["name"],
        calculate_n=gen_int_var_product_str(func_attrs["inputs"][0].shape()),
        index_type=backend_spec.index_type,
        indent=indent,
    )
