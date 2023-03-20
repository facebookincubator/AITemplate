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
Define masked_select codegen and CUDA kernel
"""
import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda import cuda_common


header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
"""

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    {{input_type}}* /*output*/,
    const {{input_type}}* /*input*/,
    const bool* /*mask*/,
    {{index_type}} /*num_elems*/,
    {{index_type}}* /*output size*/,
    void* workspace /*workspace*/,
    cudaStream_t /*stream*/
    );
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

#ifndef CUDA_CHECK_MASKED_SELECT
#define CUDA_CHECK_MASKED_SELECT(expr, msg)                   \\
  do {                                                        \\
    cudaError_t status = (expr);                              \\
    if (status != cudaSuccess) {                              \\
        std::cerr << msg << " at " << __FILE__                \\
                  << ": " << __LINE__ << std::endl;           \\
        throw std::runtime_error(cudaGetErrorString(status)); \\
    }                                                         \\
  } while (0)
#endif // CUDA_CHECK_MASKED_SELECT

void {{func_name}}(
    {{input_type}}* output,
    const {{input_type}}* input,
    const bool* mask,
    {{index_type}} num_elems,
    {{index_type}}* num_nonmasked,
    void* workspace,
    cudaStream_t stream
    ) {

    // Make sure input, output, mask, and workspace are valid
    if (!input) {
        throw std::runtime_error("input is NULL!");
    }
    if (!output) {
        throw std::runtime_error("output is NULL!");
    }
    if (!mask) {
        throw std::runtime_error("mask is NULL!");
    }
    if (!workspace) {
        throw std::runtime_error("workspace is NULL!");
    }
    size_t allocated_storage = {{workspace_size}};

    // Keep the number of nonmasked elements at the beginning of the workspace
    const size_t NUM_NONMASKED_SIZE = sizeof({{index_type}});
    {{index_type}}* num_nonmasked_device = static_cast<{{index_type}}*>(workspace);

    // Get needed temporary storage size and reallocate if necessary
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK_MASKED_SELECT(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, mask, output, num_nonmasked_device, num_elems, stream),
                             "Error when checking the required buffer size!");
    CUDA_CHECK_MASKED_SELECT(cudaStreamSynchronize(stream), "Error when synchronizing the stream!");

    if (allocated_storage < temp_storage_bytes + NUM_NONMASKED_SIZE) {
        auto msg = "Got pre-allocated buffer of size " + std::to_string(allocated_storage) + ", but need " + std::to_string(temp_storage_bytes)
                + ". Allocating a new buffer, expect performance degradation.";
        std::cerr << msg << std::endl;
        // Allocate temporary storage
        temp_storage_bytes += NUM_NONMASKED_SIZE;
        CUDA_CHECK_MASKED_SELECT(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream), "Error when trying to allocate a new buffer!");
        CUDA_CHECK_MASKED_SELECT(cudaStreamSynchronize(stream), "Error when synchronizing the stream!");
        workspace = d_temp_storage;
        allocated_storage = temp_storage_bytes;
    }
    allocated_storage -= NUM_NONMASKED_SIZE;  // First NUM_NONMASKED_SIZE bytes are reserved

    // Select nonmasked elements. First NUM_NONMASKED_SIZE bytes of workspace are reserved for num_nonmasked_device
    CUDA_CHECK_MASKED_SELECT(cub::DeviceSelect::Flagged(workspace + NUM_NONMASKED_SIZE, allocated_storage, input, mask, output,
        num_nonmasked_device, num_elems, stream),  "Error when selecting nonmasked elements!");

    // Extract number of nonmasked elements (size of the output)
    CUDA_CHECK_MASKED_SELECT(cudaMemcpyAsync(num_nonmasked, num_nonmasked_device, NUM_NONMASKED_SIZE, cudaMemcpyDeviceToHost, stream),
                             "Error when copying the number of nonmasked elements from device to host!");
    CUDA_CHECK_MASKED_SELECT(cudaStreamSynchronize(stream), "Error when synchronizing the stream!");

    if (d_temp_storage != nullptr) {
        CUDA_CHECK_MASKED_SELECT(cudaFreeAsync(d_temp_storage, stream), "Error when freeing GPU memory allocated by masked_select!");
    }
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}
{{indent}}  const {{index_type}} input_dims[] = {{input_dims}};
{{indent}}  int64_t num_elems = 1;
{{indent}}  for ({{index_type}} i = 0; i < {{rank}}; i++) {
{{indent}}        num_elems *= input_dims[i];
{{indent}}  }
{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{input_ptr}},
{{indent}}      {{mask_ptr}},
{{indent}}      num_elems,
{{indent}}      {{num_nonmasked}},
{{indent}}      global_workspace_,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


@registry.reg("cuda.masked_select.gen_function")
def gen_function(func_attrs) -> str:
    """
    Generate function body

    Returns
    -------
    str
        The function body string
    """
    backend_spec = CUDASpec()
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    output_type = cuda_common.dtype_to_cuda_type(y._attrs["dtype"])

    if input_type != output_type:
        raise TypeError("input type must equal to output type")

    return SRC_TEMPLATE.render(
        input_type=input_type,
        index_type=backend_spec.index_type,
        func_name=func_attrs["name"],
        header_files=header_files,
        workspace_size=func_attrs["workspace"],
    )


@registry.reg("cuda.masked_select.func_decl")
def gen_function_decl(func_attrs) -> str:
    """
    Generate function declaration.

    Returns
    -------
    str
        The function declaration string
    """
    backend_spec = CUDASpec()
    x = func_attrs["inputs"][0]
    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_type=input_type,
        index_type=backend_spec.index_type,
    )


@registry.reg("cuda.masked_select.func_call")
def gen_function_call(func_attrs, indent="  ") -> str:
    """
    Generate function call.

    Returns
    -------
    str
        The function call string
    """
    backend_spec = CUDASpec()
    x, mask = func_attrs["inputs"]
    y = func_attrs["outputs"][0]

    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])

    input_ptr = backend_spec.cast_to_ptr_template.render(
        name=x._attrs["name"],
        dtype=dtype,
    )
    output_ptr = backend_spec.cast_to_ptr_template.render(
        name=y._attrs["name"],
        dtype=dtype,
    )
    mask_ptr = backend_spec.cast_to_ptr_template.render(
        name=mask._attrs["name"],
        dtype="bool",
    )
    # Number of nonmasked elements, i.e. size of the output
    num_nonmasked_ptr = "&" + y._attrs["shape"][0]._attrs["name"]
    input_dims = "{" + ",".join([dim._attrs["name"] for dim in x._attrs["shape"]]) + "}"
    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        input_name=x._attrs["name"],
        num_nonmasked=num_nonmasked_ptr,
        input_dims=input_dims,
        rank=len(x._attrs["shape"]),
        output_ptr=output_ptr,
        input_ptr=input_ptr,
        mask_ptr=mask_ptr,
        index_type=backend_spec.index_type,
    )
