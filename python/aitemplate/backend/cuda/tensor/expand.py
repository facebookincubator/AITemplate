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
from aitemplate.backend.cuda.tensor import expand_static_shape  # noqa: F401


def _to_cuda_dtype(dtype):
    dtype = CUDASpec().dtype_to_backend_dtype.get(dtype, None)
    return dtype


@registry.reg("cuda.expand.func_decl")
def gen_function_decl(func_attrs):
    if func_attrs["optimize_fixed_dims"] and func_attrs["non_head_dims_are_fixed"]:
        func = registry.get("cuda.expand.static.func_decl")
        return func(func_attrs)
    x = func_attrs["inputs"][0]
    func_name = func_attrs["name"]
    index_type = _to_cuda_dtype(func_attrs.get("index_type", "int64"))
    dt = x.dtype()
    dtype = _to_cuda_dtype(dt)
    assert (
        dtype is not None
    ), f"CUDA implementation does not support dtype {x.dtype()} (yet)"
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_name,  # name of the function
        dtype=dtype,  # data type of the input and output tensor elements ( valid CUDA C type like float ))
        index_type=index_type,
    )


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void* src,
  const {{index_type}}* input_dims,
  const {{index_type}} input_rank,
  void* dst,
  {{index_type}}* output_dims, // written to ( runtime shape inference )
  const {{index_type}} output_rank,
  const {{index_type}}* output_dim_types,
  cudaStream_t stream);
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <limits>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "logging.h"

using bfloat16 = __nv_bfloat16;

{% if index_type=="int64_t" %}
#define DIM_TYPE_ADD 0l
#define DIM_TYPE_EXPAND 1l
#define DIM_TYPE_KEEP 2l

#define MAX_THREADS_PER_BLOCK 1024l
#define MAX_BLOCKS 65535l
#define MAX_X_BLOCKS 2147483647l
{% else %}
#define DIM_TYPE_ADD 0
#define DIM_TYPE_EXPAND 1
#define DIM_TYPE_KEEP 2

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 65535
#define MAX_X_BLOCKS 2147483647
{% endif %}

// integer ceil division
#define INT_CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

/**
 * Sequential write expand kernel for single block case.
 *
 * This kernel is optimized for small inputs, where we can load
 * the entire  input into shared memory more or less at once
 */
__global__ void {{func_name}}_sequential_write_single_block_kernel(
  // Implementation for small inputs where the entire src can be read into shared memory,
  // and we have just one thread block
  const {{dtype}}* src,
  const {{index_type}} src_numel,
  {{dtype}}* dst,
  const {{index_type}} dst_numel
  {% for i in range(output_rank) %}
        ,const {{index_type}} output_strides_{{i}}
        ,const {{index_type}} read_strides_{{i}}
  {% endfor %}
  ) {
    // determine our range of elements to read
    const {{index_type}} write_idx = threadIdx.x;
    extern __shared__ {{dtype}} src_shared[]; // dynamic shared memory
    if (write_idx<src_numel) {
        src_shared[write_idx] = src[write_idx];
    }
    __syncthreads();
    {{index_type}} read_idx = 0;
    {{index_type}} remaining_idx = write_idx; // Used to calculate remainder
    {% for i in range(output_rank) %}
        read_idx += (remaining_idx / output_strides_{{i}}) * read_strides_{{i}};
        remaining_idx %= output_strides_{{i}};
    {% endfor %}
    if (write_idx<dst_numel) {
        dst[write_idx] = src_shared[read_idx];
    }
}

/**
 * Sequential write expand kernel with batched read/writes on trailing
 * dimensions.
 *
 * This kernel is optimized for the case that trailing dimensions
 * are kept between input and output, in which case we can do block-wise
 * reads and writes.
 */
__global__ void {{func_name}}_sequential_write_batch_kernel(

  const {{dtype}}* src,
  {{dtype}}* dst,
  const {{index_type}} dst_numel,
  const {{index_type}} batch_size
  {% for i in range(output_rank) %}
        ,const {{index_type}} output_strides_{{i}}
        ,const {{index_type}} read_strides_{{i}}
  {% endfor %}
  ) {
    // determine our range of elements to read
    const {{index_type}} write_idx = (blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + blockDim.z * blockIdx.z + threadIdx.x) * batch_size;
    {{index_type}} read_idx = 0;
    {{index_type}} i = write_idx; // Used to calculate remainder
    {% for i in range(output_rank) %}
        read_idx += (i / output_strides_{{i}}) * read_strides_{{i}};
        i %= output_strides_{{i}};
    {% endfor %}
    if (write_idx+batch_size-1<dst_numel) {
        dst[write_idx] = src[read_idx];
        for (i = 1; i < batch_size; i++) {
            dst[write_idx+i] = src[read_idx+i];
        }
    }
}

/**
 * Sequential write expand kernel.
 * This kernel deals with the general case. It relies heavily on L2 cache
 * for scattered read optimization and does sequential writes.
 * This was benchmarked against an alternative implementation that tried
 * to minimize overall memory accesses, doing sequential reads and scattered
 * writes. But this implementation is faster.
 */
__global__ void {{func_name}}_sequential_write_kernel(

  const {{dtype}}* src, // source tensor
  {{dtype}}* dst, // destination tensor
  const {{index_type}} dst_numel // number of elements in dst
  {% for i in range(output_rank) %}
        ,const {{index_type}} output_strides_{{i}} // Stride for writing dimension {{i}} to dst
        ,const {{index_type}} read_strides_{{i}} // Stride for reading dimension {{i}} from src
  {% endfor %}
  ) {
    // determine our range of elements to read
    const {{index_type}} write_idx = blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + blockDim.z * blockIdx.z + threadIdx.x;
    {{index_type}} read_idx = 0;
    {{index_type}} remaining_idx = write_idx; // Used to calculate remainder
    {% for i in range(output_rank) %}
        read_idx += (remaining_idx / output_strides_{{i}}) * read_strides_{{i}};
        remaining_idx %= output_strides_{{i}};
    {% endfor %}
    if (write_idx<dst_numel) {
        dst[write_idx] = src[read_idx];
    }
}

/**
 * Expand Operator entry point with support for dynamic shapes
 */
void {{func_name}} (
  const void* src, // input tensor
  const {{index_type}}* input_dims, // input dimensions ( passed by value )
  const {{index_type}} input_rank,
  void* dst, // output tensor
  {{index_type}}* output_dims, // output dimensions ( passed by value )
  const {{index_type}} output_rank,
  const {{index_type}}* output_dim_types, // Output dim types ( length=output_rank ). 2 = keep dimension, 1 = expand dimension, 0 = add dimension
  cudaStream_t stream)
{
  // Calculate number of input elements
  {{index_type}} input_numel = 1;
  {{index_type}} i;
  for (i = 0; i < input_rank; ++i) {
    input_numel *= input_dims[i];
  }
  {{index_type}} input_dim_pos = 0;

  // Calculate number of output dimensions
  {{index_type}} output_numel = 1;
  for (i = 0; i < output_rank; ++i) {
    output_numel *= output_dims[i];
  }

  // Determine stride for each input dimension
  {{index_type}} input_strides[input_rank];
  input_strides[input_rank-1] = 1;
  for (i=input_rank-2;i>=0;--i) {
    input_strides[i] = input_strides[i+1]*input_dims[i+1];
  }
  // Determine stride for each output dimension
  {{index_type}} output_strides[output_rank];
  output_strides[output_rank-1] = 1;
  for (i=output_rank-2;i>=0;--i) {
    output_strides[i] = output_strides[i+1]*(output_dims[i+1]);
  }

  // Determine read strides for each output dimension
  // (0 for expand or add dims, otherwise the stride of
  // of the corresponding input dim)
  {{index_type}} read_strides[output_rank];

  input_dim_pos = 0;
  for (i = 0; i < output_rank; ++i) {
    {{index_type}} dim_type =  output_dim_types[i];
    if (dim_type == DIM_TYPE_KEEP ) { // keep
      read_strides[i] = input_strides[input_dim_pos++];
    } else {
      read_strides[i] = 0;
      if (dim_type==DIM_TYPE_EXPAND) {
        input_dim_pos++;
      }
    }
  }
  assert(input_dim_pos==input_rank);

  // Calculating tail dimension in order to determine whether we can do sequential batching
  {{index_type}} tail_dim = 1;
  for (i = output_rank-1; i >= 0; --i) {
      if (output_dim_types[i]!=DIM_TYPE_KEEP) {
         break;
      }
      tail_dim *= output_dims[i];
  }

  {{index_type}} batch_size = 1; // sequential batch len

  if (output_numel>MAX_THREADS_PER_BLOCK) {
    // If the input/output is so small that we can read it all into shared mem,
    // sequential batching makes no sense
    batch_size = 7; // Determined experimentally via benchmark.
                    // Should be reevaluated after algorithmic changes.
    for (;batch_size>1;--batch_size) {
      if ((tail_dim % batch_size)==0) {
          break;
      }
    }
  }
  assert ((output_numel % batch_size)==0);

  // determine CUDA kernel grid layout
  {{index_type}} output_batches = output_numel / batch_size;

  {{index_type}} block_size = INT_CEIL_DIV(output_batches, MAX_THREADS_PER_BLOCK);
  {{index_type}} thread_size_x = min(MAX_THREADS_PER_BLOCK, output_batches);

  {{index_type}} block_size_x = block_size;
  {{index_type}} block_size_y = 1;
  {{index_type}} block_size_z = 1;

  // for very large dimensions, we need to split into x,y,z grid blocks
  if (block_size_x>MAX_X_BLOCKS) {
      block_size_y = INT_CEIL_DIV(block_size_x, MAX_X_BLOCKS);
      block_size_x = MAX_X_BLOCKS;
      if (block_size_y > MAX_BLOCKS) {
        block_size_z = INT_CEIL_DIV(block_size_y, MAX_BLOCKS);
        block_size_y = MAX_BLOCKS;
      }
  }
  dim3 dimGrid(block_size_x, block_size_y, block_size_z);
  dim3 dimBlock(thread_size_x, 1, 1);
  // Select the right kernel to call and call it
  if (batch_size==1) {
    if (block_size_x>1) {
      {{func_name}}_sequential_write_kernel<<<dimGrid,dimBlock,0,stream>>>(
          static_cast<const {{dtype}}*>(src),
          static_cast<{{dtype}}*>(dst),
          output_numel
          {% for i in range(output_rank) %}
            ,output_strides[{{i}}]
            ,read_strides[{{i}}]
          {% endfor %}
      );
    } else {
      {{func_name}}_sequential_write_single_block_kernel<<<dimGrid,dimBlock,sizeof({{dtype}})*input_numel,stream>>>(
          static_cast<const {{dtype}}*>(src),
          input_numel,
          static_cast<{{dtype}}*>(dst),
          output_numel
          {% for i in range(output_rank) %}
            ,output_strides[{{i}}]
            ,read_strides[{{i}}]
          {% endfor %}
      );
    }
  } else {  // batch_size>1, asserting (thread_size_x % batch_size)==0
      {{func_name}}_sequential_write_batch_kernel<<<dimGrid,dimBlock,0,stream>>>(
          static_cast<const {{dtype}}*>(src),
          static_cast<{{dtype}}*>(dst),
          output_numel,
          batch_size
          {% for i in range(output_rank) %}
            ,output_strides[{{i}}]
            ,read_strides[{{i}}]
          {% endfor %}
      );
  }
}
"""
)


def create_template_args(func_attrs: Dict[str, Any], indent="  "):
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    dst = y._attrs["name"]
    src = x._attrs["name"]
    func_name = func_attrs["name"]
    dtype = _to_cuda_dtype(x.dtype())
    assert (
        dtype is not None
    ), f"CUDA implementation does not support dtype {x.dtype()} (yet)"

    xshape = x._attrs["shape"]
    yshape = y._attrs["shape"]
    index_type = _to_cuda_dtype(func_attrs.get("index_type", "int64"))
    assert index_type is not None

    input_dims = ",".join(
        [f"static_cast<{index_type}>(" + dim._attrs["name"] + ")" for dim in xshape]
    )
    output_dims = ",".join(
        [f"static_cast<{index_type}>(" + dim._attrs["name"] + ")" for dim in yshape]
    )
    input_rank = len(xshape)
    output_rank = len(yshape)
    dim_types = ",".join([str(int(dt)) for dt in func_attrs["dim_types"]])
    return {
        "func_name": func_name,  # name of the function
        "dst": dst,  # name of the output tensor (of type dtype*)
        "src": src,  # name of the input tensor (of type dtype*)
        "input_dims": input_dims,  # list of input dimensions (as string of comma-separated variable names )
        "output_dims": output_dims,  # output dimensions (as string of comma-separated variable names)
        "input_rank": input_rank,  # number of input dimensions
        "output_rank": output_rank,  # number of output dimensions
        "dim_types": dim_types,  # list of output dimension types: 2 = keep, 1 = expand, 0 = add
        "dtype": dtype,  # data type of the input and output tensor elements ( valid CUDA C type like float ))
        "indent": indent,  # indentation for the function call template,
        "index_type": index_type,
    }


@registry.reg("cuda.expand.gen_function")
def gen_function(func_attrs):
    if not (
        func_attrs["optimize_fixed_dims"] and func_attrs["non_head_dims_are_fixed"]
    ):
        return SRC_TEMPLATE.render(create_template_args(func_attrs, "    "))
    else:
        func = registry.get("cuda.expand.static.gen_function")
        return func(func_attrs)


@registry.reg("cuda.expand.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    if not (
        func_attrs["optimize_fixed_dims"] and func_attrs["non_head_dims_are_fixed"]
    ):
        return FUNC_CALL_TEMPLATE.render(create_template_args(func_attrs, indent))
    else:
        func = registry.get("cuda.expand.static.func_call")
        return func(func_attrs, indent)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
    {
    {{indent}}const {{index_type}} input_dims[] = { {{input_dims}} };
    {{indent}}{{index_type}} output_dims[] = { {{output_dims}} };
    {{indent}}const {{index_type}} output_dim_types[] = { {{dim_types}} };
    {{indent}}{{func_name}}(
    {{indent}}    {{src}},
    {{indent}}    input_dims,
    {{indent}}    {{input_rank}},
    {{indent}}    {{dst}},
    {{indent}}    output_dims,
    {{indent}}    {{output_rank}},
    {{indent}}    output_dim_types,
    {{indent}}    stream);
    }
    """
)
