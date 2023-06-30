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
Codegen functions for the jagged_lengths_to_offsets op.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec


SRC_TEMPLATE = jinja2.Template(
    """
#include <stdexcept>

#include <cub/cub.cuh>

void {{func_name}}(
    const void* lengths,
    void* offsets,
    {{index_type}} batch_size,
    {{index_type}}* offsets_size,
    void* workspace,
    cudaStream_t stream
) {
    *offsets_size = batch_size + 1;

    // pre-initialize all offset values to zero
    cudaMemsetAsync(offsets, 0, (*offsets_size) * sizeof({{offsets_type}}), stream);

    // no-op call to determine the temp storage size;
    // although we don't need this call (because the workspace
    // is pre-allocated to a sufficiently large size), unless
    // the exact size determined by it is passed to the
    // following call, it won't perform any computation
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr,
        temp_storage_bytes,
        reinterpret_cast<const {{offsets_type}}*>(lengths),
        reinterpret_cast<{{offsets_type}}*>(offsets) + 1,
        (int)batch_size,
        stream
    );

    if (temp_storage_bytes > {{workspace_size}}) {
        throw std::runtime_error("Pre-allocated workspace size ({{workspace_size}} bytes) is too small.");
    }

    // compute the actual offsets, starting from the offsets[1]
    cub::DeviceScan::InclusiveSum(
        workspace,
        temp_storage_bytes,
        reinterpret_cast<const {{offsets_type}}*>(lengths),
        reinterpret_cast<{{offsets_type}}*>(offsets) + 1,
        (int)batch_size,
        stream
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
    void*,            /* offsets */
    {{index_type}},   /* batch_size */
    {{index_type}}*,  /* offsets_size */
    void*,            /* workspace */
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
{{indent}}    {{offsets}},
{{indent}}    {{batch_size}},
{{indent}}    &{{offsets_size}},
{{indent}}    global_workspace_,
{{indent}}    stream
{{indent}});
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


@registry.reg("cuda.jagged_lengths_to_offsets.gen_function")
def jagged_lengths_to_offsets_gen_function(func_attrs):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    offsets = func_attrs["outputs"][0]
    offsets_type = backend_spec.dtype_to_backend_type(offsets.dtype())

    return SRC_TEMPLATE.render(
        func_name=func_name,
        index_type=backend_spec.index_type,
        offsets_type=offsets_type,
        workspace_size=func_attrs["workspace"],
    )


@registry.reg("cuda.jagged_lengths_to_offsets.func_decl")
def jagged_lengths_to_offsets_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        index_type=backend_spec.index_type,
    )


@registry.reg("cuda.jagged_lengths_to_offsets.func_call")
def jagged_lengths_to_offsets_gen_function_call(func_attrs, indent="  "):
    func_name = func_attrs["name"]
    lengths = func_attrs["inputs"][0]
    offsets = func_attrs["outputs"][0]
    batch_size = lengths.shape()[0]
    offsets_size = offsets.shape()[0]

    return FUNC_CALL_TEMPLATE.render(
        indent="      ",
        func_name=func_name,
        lengths=lengths._attrs["name"],
        offsets=offsets._attrs["name"],
        batch_size=batch_size._attrs["name"],
        offsets_size=offsets_size._attrs["name"],
    )
