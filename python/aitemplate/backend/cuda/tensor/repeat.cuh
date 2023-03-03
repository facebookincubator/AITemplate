/*
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

Functions for repeating parts of a CUDA source tensor onto itself
or into a target tensor.

Used by expand_static_shape.py ( expand operator )

*/

#define INT_CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define SHM_MAX 1024 * 44

__global__ void repeat_head_kernel(
    const int64_t* const src,
    int64_t* data,
    size_t head_mem_num_elements,
    size_t num_repeat_copies) {
  extern __shared__ int64_t shared[];
  const size_t stride_y = blockDim.y * gridDim.y;
  const size_t stride_x = blockDim.x * gridDim.x;

  // outer grid-stride loop
  for (size_t ri = blockDim.x * blockIdx.x + threadIdx.x;
       ri < head_mem_num_elements;
       ri += stride_x) {
    // read only with one thread per y dim
    if (threadIdx.y == 0) {
      // in y direction: thread 0 reads, all threads write
      // repeatedly direct async copy from global to shared memory, see
      // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#optimizing-cuda-applications
      // and
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memcpy-async-primitiv
      __pipeline_memcpy_async(&shared[threadIdx.x], &src[ri], sizeof(int64_t));
      __pipeline_commit();
      __pipeline_wait_prior(0);
    }
    __syncthreads(); // wait for shared memory to be populated
    // inner grid-stride loop, write with all threads out of shared memory
    size_t wi = threadIdx.y + blockDim.y * blockIdx.y;
    for (; wi < num_repeat_copies; wi += stride_y) {
      data[ri + head_mem_num_elements * wi] = shared[threadIdx.x];
    }
  }
}

__host__ cudaError_t cuda_repeat_head_vectorized(
    const int64_t* const src,
    int64_t* data,
    size_t head_mem_num_elements,
    size_t num_repeat_copies,
    cudaStream_t stream) {
  size_t threads_x = 32;
  size_t threads_y = 1024 / threads_x;
  size_t blocks_x = INT_CEIL_DIV(head_mem_num_elements, threads_x);
  size_t blocks_y = INT_CEIL_DIV(num_repeat_copies, threads_y);
  size_t serialization_level =
      INT_CEIL_DIV(threads_x * sizeof(int64_t) * blocks_x * blocks_y, SHM_MAX);
  // reduce number of blocks if necessary, so we do not exceed available shared
  // memory
  blocks_y = INT_CEIL_DIV(
      blocks_y, serialization_level); // reduce thread count in y dimension
                                      // first, e.g. sequentialized writes
  serialization_level =
      INT_CEIL_DIV(threads_x * sizeof(int64_t) * blocks_x * blocks_y, SHM_MAX);
  // reduce number of blocks in x direction if this is not sufficient yet
  blocks_x = INT_CEIL_DIV(blocks_x, serialization_level);
  dim3 dimGrid(blocks_x, blocks_y);
  dim3 dimBlock(threads_x, threads_y);
  repeat_head_kernel<<<
      dimGrid,
      dimBlock,
      threads_x * sizeof(int64_t),
      stream>>>(src, data, head_mem_num_elements, num_repeat_copies);
  return cudaPeekAtLastError();
}

__host__ cudaError_t cuda_repeat_head(
    void* data,
    const size_t head_mem_bytes,
    size_t num_repeat_copies,
    cudaStream_t stream) {
  cudaError_t res = cudaSuccess;
  if (num_repeat_copies == 0)
    return res;
  if ((head_mem_bytes % 8) == 0) {
    // no need to double memory any further if it is 64-bit aligned
    res = cuda_repeat_head_vectorized(
        static_cast<const int64_t* const>(data),
        static_cast<int64_t*>(data) + (head_mem_bytes / 8),
        head_mem_bytes / 8,
        num_repeat_copies,
        stream);
    if (res != cudaSuccess) {
      return res;
    }
  } else {
    res = cudaMemcpyAsync(
        static_cast<void*>(static_cast<uint8_t*>(data) + head_mem_bytes),
        data,
        head_mem_bytes,
        cudaMemcpyDeviceToDevice,
        stream);
    if (res != cudaSuccess) {
      return res;
    }
    if (num_repeat_copies >= 2) {
      // recurse
      // we have already repeated 1 time, therefore the (num_repeat_copies-1)
      res = cuda_repeat_head(
          data, head_mem_bytes * 2, (num_repeat_copies - 1) / 2, stream);
      if (res != cudaSuccess) {
        return res;
      }
      // deal with possible remainder
      if (((num_repeat_copies - 1) % 2) == 1) {
        res = cudaMemcpyAsync(
            static_cast<void*>(
                static_cast<uint8_t*>(data) +
                num_repeat_copies * head_mem_bytes),
            data,
            head_mem_bytes,
            cudaMemcpyDeviceToDevice,
            stream);
      }
    }
  }
  return res;
}

__host__ cudaError_t cuda_repeat_src(
    const void* const src,
    void* data,
    const size_t head_mem_bytes,
    size_t num_repeat_copies,
    cudaStream_t stream) {
  cudaError_t res = cudaSuccess;
  if (num_repeat_copies == 0) {
    return res;
  }

  res = cudaMemcpyAsync(
      data, src, head_mem_bytes, cudaMemcpyDeviceToDevice, stream);
  if ((res != cudaSuccess) || (num_repeat_copies == 1)) {
    return res;
  }
  return cuda_repeat_head(data, head_mem_bytes, num_repeat_copies - 1, stream);
}
