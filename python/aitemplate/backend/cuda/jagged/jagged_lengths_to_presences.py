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
Codegen functions for the jagged_lengths_to_presences op.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec


SRC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using bfloat16 = nv_bfloat16;

#define THREADS_PER_BLOCK 128


namespace {

__global__ void jagged_lengths_to_presences_kernel(
    const {{lengths_type}}* lengths,
    {{presences_type}}* presences
) {
    {{index_type}} bid = blockIdx.y;
    {{index_type}} tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < {{max_seq_len}}) {
        {{lengths_type}} len = lengths[bid];
        presences[bid * {{max_seq_len}} + tid] = static_cast<{{presences_type}}>(tid < len);
    }
}

} // namespace


void {{func_name}}(
    const void* lengths,
    void* presences,
    {{index_type}} batch_size,
    cudaStream_t stream
) {
    dim3 grid_size(({{max_seq_len}} + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, batch_size);
    jagged_lengths_to_presences_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const {{lengths_type}}*>(lengths),
        reinterpret_cast<{{presences_type}}*>(presences)
    );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    const void*,      /* lengths */
    void*,            /* presences */
    {{index_type}},   /* batch_size */
    cudaStream_t      /* stream */
);
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{lengths}},
{{indent}}    {{presences}},
{{indent}}    {{batch_size}},
{{indent}}    stream
{{indent}});
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


@registry.reg("cuda.jagged_lengths_to_presences.gen_function")
def jagged_lengths_to_presences_gen_function(func_attrs):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    lengths = func_attrs["inputs"][0]
    presences = func_attrs["outputs"][0]
    lengths_type = backend_spec.dtype_to_backend_type(lengths.dtype())
    presences_type = backend_spec.dtype_to_backend_type(presences.dtype())
    max_seq_len = presences.shape()[1].value()

    return SRC_TEMPLATE.render(
        func_name=func_name,
        lengths_type=lengths_type,
        presences_type=presences_type,
        index_type=backend_spec.index_type,
        max_seq_len=max_seq_len,
    )


@registry.reg("cuda.jagged_lengths_to_presences.func_decl")
def jagged_lengths_to_presences_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        index_type=backend_spec.index_type,
    )


@registry.reg("cuda.jagged_lengths_to_presences.func_call")
def jagged_lengths_to_presences_gen_function_call(func_attrs, indent="  "):
    func_name = func_attrs["name"]
    lengths = func_attrs["inputs"][0]
    presences = func_attrs["outputs"][0]
    batch_size = lengths.shape()[0]

    return FUNC_CALL_TEMPLATE.render(
        indent="      ",
        func_name=func_name,
        lengths=lengths._attrs["name"],
        presences=presences._attrs["name"],
        batch_size=batch_size._attrs["name"],
    )
