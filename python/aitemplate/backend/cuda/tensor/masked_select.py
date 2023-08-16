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
from typing import List

import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec

from aitemplate.backend.common.elementwise_common import (
    gen_dynamic_dim_str,
    get_dynamic_dims,
    get_stride_expressions,
)
from aitemplate.backend.cuda import cuda_common
from aitemplate.compiler.base import IntImm, IntVar

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
    {% if need_broadcast %} {{dynamic_dims_decl}} {% endif %}
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

{% if need_broadcast_input or need_broadcast_mask %}
__global__ void expand_input_mask_kernel(
    {% if need_broadcast_input %}
    {{input_type}}* expanded_input,
    const {{input_type}}* input,
    {% endif %}
    {% if need_broadcast_mask %}
    bool* expanded_mask,
    const bool* mask,
    {% endif %}
    {{dynamic_dims_decl}}
    const {{index_type}} num_elems
) {
    for(auto idx = blockIdx.x*blockDim.x + threadIdx.x; idx <= num_elems; idx+=gridDim.x*blockDim.x) {

        if (idx < num_elems) {

        {% if need_broadcast_input %}
            {{index_type}} input_idx = 0;
        {% endif %}
        {% if need_broadcast_mask %}
            {{index_type}} mask_idx = 0;
        {% endif %}
            {{index_type}} cur;
            auto tmp = idx;

        {% for i in range(max_rank) %}
            cur = tmp % {{max_dims[max_rank-i-1]}};
            tmp = tmp / {{max_dims[max_rank-i-1]}};
        {% if need_broadcast_input and (i < input_rank) %}
            if ({{input_dims[input_rank-i-1]}} > 1) {
                input_idx += cur * {{input_strides[input_rank-i-1]}};
            }
        {% endif %}
        {% if need_broadcast_mask and (i < mask_rank) %}
            if ({{mask_dims[mask_rank-i-1]}} > 1) {
                mask_idx += cur * {{mask_strides[mask_rank-i-1]}};
            }
        {% endif %}
        {% endfor %}

        {% if need_broadcast_input %}
            expanded_input[idx] = input[input_idx];
        {% endif %}
        {% if need_broadcast_mask %}
            expanded_mask[idx] = mask[mask_idx];
        {% endif %}
        }
    }
}
{% endif %}

void {{func_name}}(
    {{input_type}}* output,
    const {{input_type}}* input,
    const bool* mask,
    {% if need_broadcast_input or need_broadcast_mask %}
    {{dynamic_dims_decl}}
    {% endif %}
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
    constexpr size_t INPUT_TYPE_SIZE = sizeof({{input_type}});
    constexpr size_t BOOL_SIZE = sizeof(bool);
    constexpr size_t INDEX_TYPE_SIZE = sizeof({{index_type}});

    {{index_type}} workspace_offset = 0;
    {{index_type}}* num_nonmasked_device = static_cast<{{index_type}}*>(workspace+workspace_offset);
    workspace_offset += INDEX_TYPE_SIZE;
    {% if need_broadcast_input %}
    {{input_type}}* expanded_input = static_cast<{{input_type}}*>(workspace+workspace_offset);
    workspace_offset += INPUT_TYPE_SIZE * num_elems;
    {% endif %}
    {% if need_broadcast_mask %}
    bool* expanded_mask = static_cast<bool*>(workspace+workspace_offset);
    workspace_offset += BOOL_SIZE * num_elems;
    {% endif %}

    // Get needed temporary storage size and reallocate if necessary
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK_MASKED_SELECT(
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
            {% if need_broadcast_input %}
            expanded_input,
            {% else %}
            input,
            {% endif %}
            {% if need_broadcast_mask %}
            expanded_mask,
            {% else %}
            mask,
            {% endif %}
            output, num_nonmasked_device, num_elems, stream),
        "Error when checking the required buffer size!"
    );
    CUDA_CHECK_MASKED_SELECT(
        cudaStreamSynchronize(stream),
        "Error when synchronizing the stream!"
    );

    if (allocated_storage < temp_storage_bytes + workspace_offset) {
        auto msg = "Got pre-allocated buffer of size " + std::to_string(allocated_storage)
            + ", but need " + std::to_string(temp_storage_bytes+workspace_offset)
            + ". Allocating a new buffer, expect performance degradation.";
        std::cerr << msg << std::endl;
        // Allocate temporary storage
        temp_storage_bytes += workspace_offset;
        CUDA_CHECK_MASKED_SELECT(
            cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream),
            "Error when trying to allocate a new buffer!"
        );
        CUDA_CHECK_MASKED_SELECT(
            cudaStreamSynchronize(stream),
            "Error when synchronizing the stream!"
        );
        workspace = d_temp_storage;
        allocated_storage = temp_storage_bytes;
    }
    allocated_storage -= workspace_offset;

    {% if need_broadcast_input or need_broadcast_mask %}
    const {{index_type}} THREADS_PER_BLOCK  = 256;
    const {{index_type}} ELEMS_PER_THREAD = 128;
    auto blocks = (num_elems + THREADS_PER_BLOCK * ELEMS_PER_THREAD) / (THREADS_PER_BLOCK * ELEMS_PER_THREAD);
    expand_input_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        {% if need_broadcast_input %}
        expanded_input,
        input,
        {% endif %}
        {% if need_broadcast_mask %}
        expanded_mask,
        mask,
        {% endif %}
        {{dynamic_dims_call}} num_elems);
    {% endif %}

    // Select nonmasked elements
    CUDA_CHECK_MASKED_SELECT(
        cub::DeviceSelect::Flagged(workspace+workspace_offset, allocated_storage,
            {% if need_broadcast_input %}
            expanded_input,
            {% else %}
            input,
            {% endif %}
            {% if need_broadcast_mask %}
            expanded_mask,
            {% else %}
            mask,
            {% endif %}
            output, num_nonmasked_device, num_elems, stream),
        "Error when selecting nonmasked elements!"
    );

    // Extract number of nonmasked elements (size of the output)
    CUDA_CHECK_MASKED_SELECT(
        cudaMemcpyAsync(num_nonmasked, num_nonmasked_device, INDEX_TYPE_SIZE, cudaMemcpyDeviceToHost, stream),
        "Error when copying the number of nonmasked elements from device to host!"
    );
    CUDA_CHECK_MASKED_SELECT(
        cudaStreamSynchronize(stream),
        "Error when synchronizing the stream!"
    );

    if (d_temp_storage != nullptr) {
        CUDA_CHECK_MASKED_SELECT(
            cudaFreeAsync(d_temp_storage, stream),
            "Error when freeing GPU memory allocated by masked_select!"
        );
    }
}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  const {{index_type}} max_dims[] = {{max_dims}};
{{indent}}  int64_t num_elems = 1;
{{indent}}  for ({{index_type}} i = 0; i < {{max_rank}}; i++) {
{{indent}}        num_elems *= max_dims[i];
{{indent}}  }
{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{input_ptr}},
{{indent}}      {{mask_ptr}},
{{indent}}      {% if need_broadcast %} {{dynamic_dims_call}} {% endif %}
{{indent}}      num_elems,
{{indent}}      {{num_nonmasked}},
{{indent}}      global_workspace_,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


def _get_dims(shape: List[IntVar]) -> List[str]:
    return [
        str(dim.value()) if isinstance(dim, IntImm) else dim._attrs["name"]
        for dim in shape
    ]


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
    x, mask = func_attrs["inputs"]
    output = func_attrs["outputs"][0]
    max_shape = func_attrs["max_shape"]

    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])
    output_type = cuda_common.dtype_to_cuda_type(output._attrs["dtype"])

    if input_type != output_type:
        raise TypeError("input type must equal to output type")

    dynamic_dims = get_dynamic_dims(x.shape(), mask.shape())

    return SRC_TEMPLATE.render(
        input_type=input_type,
        index_type=backend_spec.index_type,
        func_name=func_attrs["name"],
        header_files=header_files,
        workspace_size=func_attrs["workspace"],
        input_dims=_get_dims(x.shape()),
        input_rank=len(x.shape()),
        input_strides=get_stride_expressions(x.shape()) + ["1"],
        need_broadcast_input=x._attrs["shape"] != max_shape,
        mask_dims=_get_dims(mask.shape()),
        mask_rank=len(mask.shape()),
        mask_strides=get_stride_expressions(mask.shape()) + ["1"],
        need_broadcast_mask=mask._attrs["shape"] != max_shape,
        max_dims=_get_dims(max_shape),
        max_rank=len(max_shape),
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
    x, mask = func_attrs["inputs"]
    input_type = cuda_common.dtype_to_cuda_type(x._attrs["dtype"])

    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_type=input_type,
        index_type=backend_spec.index_type,
        need_broadcast=x._attrs["shape"] != mask._attrs["shape"],
        dynamic_dims_decl=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(x.shape(), mask.shape()),
            has_type=True,
        ),
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
    max_shape = func_attrs["max_shape"]
    # Number of nonmasked elements, i.e. size of the output
    num_nonmasked_ptr = "&" + y._attrs["shape"][0]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        input_name=x._attrs["name"],
        num_nonmasked=num_nonmasked_ptr,
        max_dims="{" + ",".join([dim._attrs["name"] for dim in max_shape]) + "}",
        max_rank=len(max_shape),
        need_broadcast=x._attrs["shape"] != mask._attrs["shape"],
        dynamic_dims_call=gen_dynamic_dim_str(
            index_type=backend_spec.index_type,
            dynamic_dims=get_dynamic_dims(x.shape(), mask.shape()),
            has_type=False,
        ),
        output_ptr=output_ptr,
        input_ptr=input_ptr,
        mask_ptr=mask_ptr,
        index_type=backend_spec.index_type,
    )
