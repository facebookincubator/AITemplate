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

import math
import os
from itertools import accumulate
from operator import mul
from typing import Any, Dict, List

import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target
from aitemplate.compiler.ops.tensor.expand import ExpandDimensionType

"""
Specialized and optimized CUDA kernel declarations for the `expand` operator
dealing with the most common case that the input and target shapes are known at compile time,
with the possible exception of leading dimensions.

"""


@registry.reg("cuda.expand.static.func_decl")
def gen_function_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(create_template_args(func_attrs))


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}} (
  const {{dtype}}* const src, // input tensor
  {{dtype}}* const dst, // output tensor
  const {{index_type}} head_size, // how many times to repeat the first part of the tensor.
  cudaStream_t stream);
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <limits>
#include <stdexcept>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "logging.h"


using bfloat16 = __nv_bfloat16;

#define DIM_TYPE_ADD 0
#define DIM_TYPE_EXPAND 1
#define DIM_TYPE_KEEP 2

#define MAX_THREADS_PER_BLOCK 1024l
// integer ceil division
#define INT_CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

{{custom_libs}}

/**
 * Get read base offset (e.g. excluding tail offset) in the middle part, given a write offset
 * into the middle part
 */
__forceinline__ __device__ {{index_type}} {{func_name}}_get_read_offset(const {{index_type}} write_offset) {
    {{index_type}} read_idx = 0;
    {{index_type}} remaining_write_idx = write_offset; // assert < {{mid_size*tail_size}} ( i.e. < mid_size*tail_size)
    {% for i in range(head_dim_count, head_dim_count+mid_dim_count-1) %}
        {% if read_strides[i]!=0 %}
    read_idx += (remaining_write_idx / {{output_strides[i]}}l) * {{read_strides[i]}}l;
        {% endif %}
        remaining_write_idx %= {{output_strides[i]}}l;
    {% endfor %}
    {% for i in range(head_dim_count+mid_dim_count-1, head_dim_count+mid_dim_count) %}
        {% if read_strides[i]!=0 %}
    read_idx += (remaining_write_idx / {{output_strides[i]}}l) * {{read_strides[i]}}l;
        {% endif %}
    {% endfor %}
    return read_idx;
}

/**
 *  Copies tail elements from a contiguous source memory region into a contiguous target memory region
 *  Using a grid-stride loop and the vectorized dtype
 *
 * see https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */
__forceinline__ __device__ void {{func_name}}_tail_copy(
        const {{dtype}} * const src, // base src tensor memory pointer
        const {{index_type}} read_offset, // base offset into src, via {{dtype}}-typed indexing
        {{dtype}} * const dst,  // base destination tensor memory pointer
        const {{index_type}} write_offset, // Base offset into dst via {{dtype}}-typed indexing
        const {{index_type}} block_thread_index,
        const {{index_type}} block_thread_count,
        const {{index_type}} copy_numel
    ) {
    for ({{index_type}} i=block_thread_index;i<copy_numel;i+=block_thread_count) {
        dst[write_offset+i] = src[read_offset+i];
    }
}


/**
 * Implement the "middle" part of the kernel, where we have to deal with non-contiguous reads/writes.
 *
 * Also utilizes grid-stride loop for efficiency and flexibility
 * see  * see https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */
__global__ void {{func_name}}_mid_kernel(

  const {{dtype}}* const src, // source tensor
  {{dtype}}* const dst // destination tensor
  ) {
    // determine our range of elements to read
    const {{index_type}} write_offset = (blockDim.x * blockIdx.x + threadIdx.x) * {{tail_size}}l;
    const {{index_type}} read_offset = {{func_name}}_get_read_offset(write_offset);
    const {{index_type}} grid_size_x = gridDim.x*blockDim.x;
    const {{index_type}} grid_size_y = gridDim.y*blockDim.y;
    const {{index_type}} thread_idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    for ({{index_type}} i=write_offset;i<{{mid_size*tail_size}}l;i+=grid_size_x) {
        {{func_name}}_tail_copy(src, read_offset, dst, write_offset, thread_idx_y, grid_size_y, {{tail_size}}l);
    }

}


__global__ void {{func_name}}_mid_kernel2(

  const {{dtype}}* const src, // source tensor
  {{dtype}}* const dst // destination tensor
  ) {
    // determine our range of elements to read
    const {{index_type}} write_offset = (blockDim.y * blockIdx.y + threadIdx.y) * {{tail_size}}l;
    const {{index_type}} read_offset = {{func_name}}_get_read_offset(write_offset);
    const {{index_type}} grid_size_x = gridDim.x*blockDim.x;
    const {{index_type}} grid_size_y = gridDim.y*blockDim.y;
    const {{index_type}} step_size_y = grid_size_y * {{tail_size}}l;
    const {{index_type}} thread_idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    for ({{index_type}} i=write_offset;i<{{mid_size*tail_size}}l;i+=step_size_y) {
        {{func_name}}_tail_copy(src, read_offset, dst, write_offset, thread_idx_x, grid_size_x, {{tail_size}}l);
    }

}

/**
 * Expand Operator entry point, optimized for static shapes. Only the head dimension may be dynamic.
 */
void {{func_name}} (
  const {{dtype}}* const src, // input tensor
  {{dtype}}* const dst, // output tensor
  const {{index_type}} head_size, // how many times to repeat the first part of the tensor.
  cudaStream_t stream)
{
  {% if mid_dim_count>0 %}
  // we have middle dimensions which involve non-contiguous reads
  // so we need to invoke the middle kernel
  dim3 dimGrid({{mid_grid_blocks_x}}, {{mid_grid_blocks_y}});
  dim3 dimBlock({{mid_grid_threads_x}}, {{mid_grid_threads_y}});
  {{func_name}}_mid_kernel2<<<dimGrid,dimBlock,0,stream>>>(src, dst);
  if (head_size>1l) {
     // now repeat copy what we already built once, multiple times into the rest of the output tensor
     cuda_repeat_head(dst, {{mid_size*tail_size}}l*sizeof({{dtype}}),head_size-1, stream);
  }
  {% else %}
    // we have no middle dimensions, so all we need to do is repeatedly copy the source multiple times
    // repeat the entire thing a dynamic number of times ( e.g. head_size times )
    cuda_repeat_src(src, dst, {{mid_size*tail_size}}l*sizeof({{dtype}}), head_size, stream);
  {% endif %}
}
"""
)

_dtype_sizes = {
    "half": 2,
    "bfloat16": 2,
    "float32": 4,
    "int64_t": 8,
    "int32_t": 4,
    "float": 4,
}

_size_dtypes = {
    2: "half",
    4: "float",
    8: "int64_t",
    16: "int4",
}


def _ceil(num):
    return int(math.ceil(num))


def create_template_args(func_attrs: Dict[str, Any], indent="  "):
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    dst = y._attrs["name"]
    src = x._attrs["name"]
    func_name = func_attrs["name"]
    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "repeat.cuh"
    )
    dtype = CUDASpec().dtype_to_backend_dtype[x.dtype()]
    assert (
        dtype is not None
    ), f"CUDA implementation does not support dtype {x.dtype()} (yet)"
    dtype2 = _size_dtypes.get(_dtype_sizes[dtype] * 2, None)
    dtype4 = _size_dtypes.get(_dtype_sizes[dtype] * 4, None)
    xshape = x._attrs["shape"]
    yshape = y._attrs["shape"]
    dim_types: List[ExpandDimensionType] = func_attrs["dim_types"]
    index_type = "int64_t"
    assert all(
        dim.lower_bound() == dim.upper_bound() for dim in xshape
    ), "All input shapes need to be fixed"
    assert all(
        dim.lower_bound() == dim.upper_bound() for dim in yshape
    ), "All output shapes need to be fixed"

    # Calculate number of times we can repeatedly copy the entire result, based on how many add, expand and singleton dimensions
    # we have at the start
    head_size_lower = 1  # Number of times we can batch-repeat the entire result in an efficient batch-copying manner
    head_size_upper = 1
    head_dim_count = 0  # Number of head dimensions

    for dim_type, dim in zip(func_attrs["dim_types"], yshape):
        if dim_type == ExpandDimensionType.KEEP_DIM and dim.lower_bound() != 1:
            break
        head_size_lower *= dim.lower_bound()
        head_size_upper *= dim.upper_bound()
        head_dim_count += 1

    # Create a symbolic term for calculating head size ( e.g. repeat count )
    if head_size_lower == head_size_upper:
        head_size_symbolic = f"{head_size_upper}l"
    else:
        head_size_symbolic = "*".join(
            [
                f"static_cast<{index_type}>(" + dim._attrs["name"] + ")"
                for dim in yshape[:head_dim_count]
            ]
        )

    # Calculate number of tail elements, e.g. number of elements we can batch-copy in the inner loop
    # via effective sequential reads & writes
    tail_dim_count = 0  # number of tail dimensions
    tail_size = 1  # Number of the elements in all these  tail dimensions
    for dim_type, dim in reversed(
        list(zip(dim_types[head_dim_count:], yshape[head_dim_count:]))
    ):
        if dim_type != ExpandDimensionType.KEEP_DIM and dim.lower_bound() != 1:
            break
        tail_dim_count += 1
        tail_size *= dim.lower_bound()

    input_strides = list(
        reversed(
            list(accumulate([1] + [d.lower_bound() for d in reversed(xshape)], mul))
        )
    )
    output_strides = list(
        reversed(
            list(
                accumulate(
                    [1] + [d.lower_bound() for d in reversed(yshape[head_dim_count:])],
                    mul,
                )
            )
        )
    )

    output_numel = output_strides[
        0
    ]  # this does not include the number of elements obtained from head repetitions
    # since we have excluded head dimensions above
    input_numel = input_strides[0]

    mid_size = output_numel // tail_size
    mid_dim_count = len(yshape) - tail_dim_count - head_dim_count

    mid_expansion_rate = mid_size * tail_size // input_numel

    # remove the first dimension, which is the total number of elements
    # and prepend the head_dims with stride 0
    output_strides = [0] * head_dim_count + output_strides[1:]
    input_strides = input_strides[1:]

    input_stride_pos = 0
    read_strides = [0] * len(yshape)
    for i in range(len(yshape)):
        if dim_types[i] == ExpandDimensionType.ADD_DIM:
            continue
        if dim_types[i] == ExpandDimensionType.KEEP_DIM:
            read_strides[i] = input_strides[input_stride_pos]
        # For keep dim, read stride remains at zero
        input_stride_pos += 1

    assert input_stride_pos == len(
        xshape
    ), "Incorrect number of keep and expand dims. Something went wrong."
    output_rank = len(yshape)
    dim_types = ",".join([str(int(dt)) for dt in func_attrs["dim_types"]])

    # If tail size is aligned to 2 or 4 elements, we can vectorize reads/writes
    # Note: Further vectorization not easily possible, given that it could happen that
    # the read offset and the write offset can get different alignments within the expand op
    #
    if (tail_size % 4 == 0) and (dtype4 is not None):
        dtype = dtype4
        tail_size = tail_size // 4
        output_strides = [s // 4 for s in output_strides]
        read_strides = [s // 4 for s in read_strides]
    elif tail_size % 2 == 0:
        dtype = dtype2
        tail_size = tail_size // 2
        output_strides = [s // 2 for s in output_strides]
        read_strides = [s // 2 for s in read_strides]

    mid_grid_blocks_x = 1
    mid_grid_threads_x = min(tail_size, 32)
    mid_max_y_threads = 1024 // mid_grid_threads_x  # guaranteed to be >= 1
    mid_grid_threads_y = min(
        mid_max_y_threads, mid_size
    )  # so that  mid_grid_threads_x*max_x_threads <= 1024
    mid_grid_blocks_y = _ceil(mid_size / mid_grid_threads_y)

    if dtype == "bfloat16":
        # bfloat16 is not available in model-generated.h as a type,
        # so we can either just declare the input to be void*
        # or  just use the fact that we don't care about how to interpret the value
        # and just treat it like every other 16 bit type.
        dtype = "half"

    return {
        "func_name": func_name,  # name of the function
        "dst": dst,  # name of the output tensor (of type dtype*)
        "src": src,  # name of the input tensor (of type dtype*)
        "output_strides": output_strides,  # list of output stride values
        "read_strides": read_strides,  # list of read stride values
        "tail_dim_count": tail_dim_count,  # number of tail dimensions
        "tail_size": tail_size,  # number of elements in all these tail dimensions
        "head_dim_count": head_dim_count,  # number of head dimensions
        "head_size": head_size_symbolic,  # number of elements in all these head dimensions
        "mid_dim_count": mid_dim_count,
        "mid_size": mid_size,
        "mid_expansion_rate": mid_expansion_rate,  # How many times do we read the input for the middle
        "output_rank": output_rank,  # number of output dimensions
        "dim_types": dim_types,  # list of output dimension types: 2 = keep, 1 = expand, 0 = add
        "dtype": dtype,  # data type of the input and output tensor elements ( valid CUDA C type like float )
        "indent": indent,  # indentation for the function call template,
        "index_type": index_type,
        "mid_grid_blocks_y": mid_grid_blocks_y,
        "mid_grid_blocks_x": mid_grid_blocks_x,
        "mid_grid_threads_y": mid_grid_threads_y,
        "mid_grid_threads_x": mid_grid_threads_x,
        "custom_libs": custom_libs,
    }


@registry.reg("cuda.expand.static.gen_function")
def gen_function(func_attrs):
    return SRC_TEMPLATE.render(create_template_args(func_attrs, "    "))


@registry.reg("cuda.expand.static.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return FUNC_CALL_TEMPLATE.render(create_template_args(func_attrs, indent))


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
    {
    {{indent}}{{func_name}}(
    {{indent}}    static_cast<{{dtype}}*>({{src}}),
    {{indent}}    static_cast<{{dtype}}*>({{dst}}),
    {{indent}}    {{head_size}},
    {{indent}}    stream);
    }
    """
)
