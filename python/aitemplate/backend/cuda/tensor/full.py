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


CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
"""


CONSTANT_TEMPLATE = jinja2.Template(
    """
#define N_THREADS_PER_BLOCK 256

const int N_ELEMENTS_PER_THREAD = sizeof({{read_t}}) / sizeof({{data_t}});
    """
)


FUNC_DECL = jinja2.Template(
    """
void invoke_{{func_name}}(
    void*,  /* output */
    {{prefix}}Stream_t  /* stream */
);
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}invoke_{{func_name}}(
{{indent}}    {{output}},
{{indent}}    stream
{{indent}});
    """
)


FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{constant}}

__global__  void full(
    {{read_type}}* output,
    {{index_type}} num_elements
) {
  const {{index_type}} idx = (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx * N_ELEMENTS_PER_THREAD >= num_elements) {
    return;
  }

  {{read_type}} tmp;
  {{data_type}}* p = reinterpret_cast<{{data_type}}*>(&tmp);

  #pragma unroll
  for (int i=0; i < N_ELEMENTS_PER_THREAD; i++) {
      p[i] = ({{data_type}}) ({{fill_value}});
  }

  output[idx] = tmp;
}

}  // namespace

void invoke_{{func_name}}(
    void* output,
    {{prefix}}Stream_t stream
){
    int grid_size = static_cast<int>(std::ceil(static_cast<double>({{num_elements}}) / N_ELEMENTS_PER_THREAD / N_THREADS_PER_BLOCK));
    full<<<grid_size, N_THREADS_PER_BLOCK, 0, stream>>>(reinterpret_cast<{{read_type}}*> (output), {{num_elements}});
}
    """
)


@registry.reg("cuda.full.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    y = func_attrs["outputs"][0]
    backend_spec = CUDASpec()

    # fill the maximum output Tensor size with the fill_value
    # any shape within the maximum bounds will be a subset
    num_elements = 1
    for dim in y.shape():
        num_elements *= dim.upper_bound()

    dtype = y.dtype()
    data_type = backend_spec.dtype_to_backend_type(dtype)
    read_type = backend_spec.get_elementwise_read_backend_type(num_elements, dtype)

    return FUNC_TEMPLATE.render(
        header_files=CUDA_HEADER_FILES,
        constant=CONSTANT_TEMPLATE.render(
            read_t=read_type,
            data_t=data_type,
        ),
        func_name=func_attrs["name"],
        read_type=read_type,
        data_type=data_type,
        index_type=backend_spec.index_type,
        fill_value=func_attrs["fill_value"],
        num_elements=num_elements,
        prefix=backend_spec.prefix,
    )


@registry.reg("cuda.full.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    backend_spec = CUDASpec()
    return FUNC_DECL.render(
        func_name=func_attrs["name"],
        prefix=backend_spec.prefix,
    )


@registry.reg("cuda.full.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=func_attrs["outputs"][0]._attrs["name"],
        indent=indent,
    )
