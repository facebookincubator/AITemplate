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

from ... import registry

from ...backend_spec import CUDASpec
from .. import cuda_common


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
    int64_t* /*output size*/,
    void* workspace, /*workspace*/
    cudaStream_t /*stream*/
    );
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

void {{func_name}}(
    {{input_type}}* output,
    const {{input_type}}* input,
    const bool* mask,
    int64_t* num_nonmasked,
    void* workspace,
    cudaStream_t stream
    ) {

    // Make sure input, output, and mask are valid
    if (!input) {
        throw std::runtime_error("input is NULL!");
    }
    if (!output) {
        throw std::runtime_error("output is NULL!");
    }
    if (!mask) {
        throw std::runtime_error("mask is NULL!");
    }
   size_t allocated_storage = {{workspace_size}};

   // Keep the number of nonmasked elements at the beginning of the workspace
   const size_t NUM_NONMASKED_SIZE = sizeof(int64_t);
   int64_t* num_nonmasked_device = static_cast<int64_t*>(workspace);

   // Get needed temporary storage size and reallocate if necessary
   void* d_temp_storage = nullptr;
   size_t temp_storage_bytes = 0;
   cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, mask, output, num_nonmasked_device, {{num_elems}}, stream);
   if (allocated_storage - NUM_NONMASKED_SIZE < temp_storage_bytes) {
    auto msg = "Got pre-allocated buffer of size " + std::to_string(allocated_storage) + ", but need " + std::to_string(temp_storage_bytes)
             + ". Alocating a new buffer, expect performance degradation.";
    std::cerr << msg << std::endl;
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    workspace = d_temp_storage;
    allocated_storage = temp_storage_bytes;
   }

   // Select nonmasked elements. First NUM_NONMASKED_SIZE bytes of workspace are reserved for num_nonmasked_device
   cub::DeviceSelect::Flagged(workspace + NUM_NONMASKED_SIZE, allocated_storage, input, mask, output, num_nonmasked_device, {{num_elems}}, stream);
   // Extract number of nonmasked elements (size of the output)
   cudaMemcpy(num_nonmasked, num_nonmasked_device, NUM_NONMASKED_SIZE, cudaMemcpyDeviceToHost);

   if (d_temp_storage != nullptr) {
    cudaFree(d_temp_storage);
   }
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{input_ptr}},
{{indent}}      {{mask_ptr}},
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
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    output_type = cuda_common.dtype_to_cuda_type(y._attrs["dtype"])

    if input_type != output_type:
        raise TypeError("input type must equal to output type")

    num_elems = y._attrs["shape"][0].upper_bound()

    return SRC_TEMPLATE.render(
        input_type=input_type,
        func_name=func_attrs["name"],
        header_files=header_files,
        num_elems=num_elems,
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
    x = func_attrs["inputs"][0]
    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_type=input_type,
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

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        input_name=x._attrs["name"],
        num_nonmasked=num_nonmasked_ptr,
        output_ptr=output_ptr,
        input_ptr=input_ptr,
        mask_ptr=mask_ptr,
    )
