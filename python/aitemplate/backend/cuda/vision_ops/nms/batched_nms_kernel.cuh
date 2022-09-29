//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
// CUDA batched_nms kernel

int const threadsPerBlock = sizeof(unsigned long long int) * 8;
#define THREADS_PER_BLOCK 256

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

int64_t* alignPtr(int64_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int64_t*)addr;
}

__device__ inline half hmax(const half a, const half b) {
#if __CUDA_ARCH__ >= 800
  return __hmax(a, b);
#else
  return a > b ? a : b;
#endif
}

__device__ inline half hmin(const half a, const half b) {
#if __CUDA_ARCH__ >= 800
  return __hmin(a, b);
#else
  return a < b ? a : b;
#endif
}

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const int offset_,
    const float threshold) {
  T offset = __float2half_rn(float(offset_));
  T left = hmax(a[0], b[0]), right = hmin(a[2], b[2]);
  T top = hmax(a[1], b[1]), bottom = hmin(a[3], b[3]);
  T width = hmax(right - left + offset, 0.f),
    height = hmax(bottom - top + offset, 0.f);
  float interS = __half2float(width) * __half2float(height);
  float Sa =
      __half2float(a[2] - a[0] + offset) * __half2float(a[3] - a[1] + offset);
  float Sb =
      __half2float(b[2] - b[0] + offset) * __half2float(b[3] - b[1] + offset);

  return interS > threshold * (Sa + Sb - interS);
}

template <typename T>
__global__ void nms_cuda(
    const int n_boxes,
    const float iou_threshold,
    const int offset,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  int blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  CUDA_2D_KERNEL_BLOCK_LOOP(col_start, blocks, row_start, blocks) {
    const int tid = threadIdx.x;

    if (row_start > col_start)
      return;

    const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ T block_boxes[threadsPerBlock * 4];
    if (tid < col_size) {
      block_boxes[tid * 4 + 0] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
      block_boxes[tid * 4 + 1] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
      block_boxes[tid * 4 + 2] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
      block_boxes[tid * 4 + 3] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
    }
    __syncthreads();

    if (tid < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + tid;
      const T* cur_box = dev_boxes + cur_box_idx * 4;

      int i = 0;
      unsigned long long int t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = tid + 1;
      }
      for (i = start; i < col_size; i++) {
        if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
          t |= 1ULL << i;
        }
      }
      dev_mask[cur_box_idx * gridDim.y + col_start] = t;
    }
  }
}

__global__ void gather_keep_from_mask(
    int64_t* keep,
    const unsigned long long* dev_mask,
    const int n_boxes) {
  const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  const int tid = threadIdx.x;

  // mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // initialize removed.
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; ++nblock) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
#pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; ++inblock) {
      const int i = i_offset + inblock;
      if (i >= n_boxes)
        break;

      // select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (tid == 0) {
          keep[i] = 1;
        }
        auto p = dev_mask + i * col_blocks;
        // remove all bboxes which overlap the candidate.
        for (int j = tid; j < col_blocks; j += blockDim.x) {
          if (j >= nblock)
            removed[j] |= p[j];
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

template <typename T>
void batched_nms_launcher(
    cudaStream_t stream,
    const int num_instance,
    const int keep_n,
    const float iou_threshold,
    const void* input,
    void* workspace,
    void* output,
    int64_t* mask) {
  cudaMemsetAsync(output, 0, num_instance * sizeof(int64_t), stream);

  int boxes_num = num_instance;
  const int offset = 1;
  const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;
  const int col_blocks_alloc = GET_BLOCKS(boxes_num, threadsPerBlock);

  dim3 blocks(col_blocks_alloc, col_blocks_alloc);
  dim3 threads(threadsPerBlock);

  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num,
      iou_threshold,
      offset,
      (const T*)input,

      (unsigned long long*)mask);

  gather_keep_from_mask<<<
      1,
      min(col_blocks, THREADS_PER_BLOCK),
      col_blocks * sizeof(unsigned long long),
      stream>>>((int64_t*)output, (const unsigned long long*)mask, boxes_num);
}
