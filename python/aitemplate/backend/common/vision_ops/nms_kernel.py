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
nms kernel template.
"""
import jinja2

KERNEL_TEMPLATE = jinja2.Template(
    """
/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// code adapted from
// https://github.com/NVIDIA/TensorRT/blob/main/plugin/common/kernels/nmsLayer.cu
//------------------------------------------------------------------------
// GPU kernel parameters.

template <
    typename Key,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__ void BlockSortKernel(
    Key* d_in, // Tile of input
    Key* d_out) // Elapsed cycle count of block scan
{
  enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };

  // Specialize BlockLoad type for our thread block (uses warp-striped loads for
  // coalescing, then transposes in shared memory to a blocked arrangement)
  typedef {{cub}}::BlockLoad<
      Key,
      BLOCK_THREADS,
      ITEMS_PER_THREAD,
      {{cub}}::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;

  // Specialize BlockRadixSort type for our thread block
  typedef {{cub}}::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD>
      BlockRadixSortT;

  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;

  // Per-thread tile items
  Key items[ITEMS_PER_THREAD];

  // Our current block's offset
  int block_offset = blockIdx.x * TILE_SIZE;

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);

  // Barrier for smem reuse
  __syncthreads();

  // Start cycle timer
  clock_t start = clock();

  // Sort keys
  BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);

  // Stop cycle timer
  clock_t stop = clock();

  // Store output in striped fashion
  {{cub}}::StoreDirectStriped<BLOCK_THREADS>(
      threadIdx.x, d_out + block_offset, items);

  // // Store elapsed clocks
  // if (threadIdx.x == 0)
  // {
  //     d_elapsed[blockIdx.x] = (start > stop) ? start - stop : stop - start;
  // }
}

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

typedef enum { NCHW = 0, NC4HW = 1, NC32HW = 2 } DLayout_t;

#define CSC(call, err)               \
  do {                               \
    {{prefix}}Error_t {{prefix}}Status = call;   \
    if ({{prefix}}Status != {{prefix}}Success) { \
      return err;                    \
    }                                \
  } while (0)

template <typename T>
struct Bbox {
  T xmin, ymin, xmax, ymax;
  Bbox(T xmin, T ymin, T xmax, T ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
  Bbox() = default;
};

// HASH
unsigned int hash(const void* array_, size_t size) {
  // Apply hashing only when debugging RPN codes.
  if (0) {
    const char* array_const;
    char* array;
    {{prefix}}MallocHost((void**)&array, size);
    {{prefix}}Memcpy(array, array_, size, {{prefix}}MemcpyDeviceToHost);
    array_const = array;
    unsigned int hash = 45599;
    for (size_t i = 0; i < size; i++) {
      unsigned int value = array_const[i];
      hash = hash * 1487 + value;
      hash = hash * 317;
      hash = hash % 105359;
    }
    return hash;
  } else {
    return 0;
  }
}

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t*)addr;
}

#define ASSERT_PARAM(exp)      \
  do {                         \
    if (!(exp))                \
      return STATUS_BAD_PARAM; \
  } while (0)

// CUB's bug workaround:
// To work properly for large batch size CUB segmented sort needs ridiculous
// workspace alignment.
const uintptr_t ALIGNMENT = 1 << 20;

// IOU
// template <typename TFloat>
// __device__ __host__ inline float IoU(const Bbox<TFloat>& a, const
// Bbox<TFloat>& b)
// {
//     TFloat left = max(a.xmin, b.xmin), right = min(a.xmax, b.xmax);
//     TFloat top = max(a.ymin, b.ymin), bottom = min(a.ymax, b.ymax);
//     TFloat width = max((TFloat)(right - left + (TFloat) 1.0), (TFloat) 0.0);
//     TFloat height = max((TFloat)(bottom - top + (TFloat) 1.0), (TFloat) 0.0);
//     TFloat interS = width * height;
//     TFloat Sa = (a.xmax - a.xmin + (TFloat) 1) * (a.ymax - a.ymin + (TFloat)
//     1); TFloat Sb = (b.xmax - b.xmin + (TFloat) 1) * (b.ymax - b.ymin +
//     (TFloat) 1); return (float) interS / (float) (Sa + Sb - interS);
// }

__device__ inline half hmax(const half a, const half b) {
{% if cuda_hmaxmin %}
#if __CUDA_ARCH__ >= 800
  return __hmax(a, b);
#else
  return a > b ? a : b;
#endif
{% else %}
  return a > b ? a : b;
{% endif %}
}

__device__ inline half hmin(const half a, const half b) {
{% if cuda_hmaxmin %}
#if __CUDA_ARCH__ >= 800
  return __hmin(a, b);
#else
  return a < b ? a : b;
#endif
{% else %}
  return a < b ? a : b;
{% endif %}
}

template <typename T>
__device__ __host__ inline float IoU(const Bbox<T>& a, const Bbox<T>& b) {
  T left = hmax(a.xmin, b.xmin), right = hmin(a.xmax, b.xmax);
  T top = hmax(a.ymin, b.ymin), bottom = hmin(a.ymax, b.ymax);
  T width = hmax(T(right - left + T(1.0)), T(0.0));
  T height = hmax(T(bottom - top + T(1.0)), T(0.0));
  float interS = __half2float(width) * __half2float(height);
  float Sa = __half2float(a.xmax - a.xmin + T(1.0)) *
      __half2float(a.ymax - a.ymin + T(1.0));
  float Sb = __half2float(b.xmax - b.xmin + T(1.0)) *
      __half2float(b.ymax - b.ymin + T(1.0));

  return interS / (Sa + Sb - interS);
}

__device__ __host__ inline float IoU(const Bbox<float>& a, const Bbox<float>& b) {
  float left = fmaxf(a.xmin, b.xmin), right = fminf(a.xmax, b.xmax);
  float top = fmaxf(a.ymin, b.ymin), bottom = fminf(a.ymax, b.ymax);
  float width = fmaxf(right - left + 1.0f, 0.0f);
  float height = fmaxf(bottom - top + 1.0f, 0.0f);
  float interS = width * height;
  float Sa = (a.xmax - a.xmin + 1.0f) * (a.ymax - a.ymin + 1.0f);
  float Sb = (b.xmax - b.xmin + 1.0f) * (b.ymax - b.ymin + 1.0f);

  return interS / (Sa + Sb - interS);
}

// NMS KERNEL FOR SMALL BATCH SIZE
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel1(
    const int propSize,
    Bbox<T_PROPOSALS> const* __restrict__ preNmsProposals,
    T_ROIS* __restrict__ afterNmsProposals,
    const int preNmsTopN,
    const float nmsThres,
    const int afterNmsTopN) {
  __shared__ bool kept_boxes[TSIZE * DIM];
  int kept = 0;
  int batch_offset = blockIdx.x * propSize;
  int max_box_idx = batch_offset + preNmsTopN;
  int batch_offset_out = blockIdx.x * afterNmsTopN;

  int flag_idx[TSIZE];
  int boxes_idx[TSIZE];
  Bbox<T_PROPOSALS> cur_boxes[TSIZE];

// initialize kept_boxes
#pragma unroll
  for (int i = 0; i < TSIZE; i++) {
    boxes_idx[i] = threadIdx.x + batch_offset + DIM * i;
    flag_idx[i] = threadIdx.x + DIM * i;

    if (boxes_idx[i] < max_box_idx) {
      cur_boxes[i] = preNmsProposals[boxes_idx[i]];
      kept_boxes[flag_idx[i]] = true;
    } else {
      kept_boxes[flag_idx[i]] = false;
      boxes_idx[i] = -1.0f;
      flag_idx[i] = -1.0f;
    }
  }

  int ref_box_idx = 0 + batch_offset;

  // remove the overlapped boxes
  while ((kept < afterNmsTopN) && (ref_box_idx < max_box_idx)) {
    Bbox<T_PROPOSALS> ref_box;
    ref_box = preNmsProposals[ref_box_idx];

#pragma unroll
    for (int i = 0; i < TSIZE; i++) {
      if (boxes_idx[i] > ref_box_idx) {
        if (IoU(ref_box, cur_boxes[i]) > nmsThres) {
          kept_boxes[flag_idx[i]] = false;
        }
      } else if (boxes_idx[i] == ref_box_idx) {
        afterNmsProposals[(batch_offset_out + kept) * 4 + 0] = ref_box.xmin;
        afterNmsProposals[(batch_offset_out + kept) * 4 + 1] = ref_box.ymin;
        afterNmsProposals[(batch_offset_out + kept) * 4 + 2] = ref_box.xmax;
        afterNmsProposals[(batch_offset_out + kept) * 4 + 3] = ref_box.ymax;
      }
    }
    __syncthreads();

    do {
      ref_box_idx++;
    } while (!kept_boxes[ref_box_idx - batch_offset] &&
             ref_box_idx < max_box_idx);

    kept++;
  }
}

// NMS KERNEL FOR LARGE BATCH SIZE
template <typename T_PROPOSALS, typename T_ROIS, int DIM, int TSIZE>
__global__ __launch_bounds__(DIM) void nmsKernel2(
    const int propSize,
    Bbox<T_PROPOSALS> const* __restrict__ proposals,
    T_ROIS* __restrict__ filtered,
    const int preNmsTopN,
    const float nmsThres,
    const int afterNmsTopN) {
  Bbox<T_PROPOSALS> const* cProposals = proposals + blockIdx.x * propSize;

  Bbox<T_PROPOSALS> t[TSIZE];
  uint64_t del = 0;

  for (int i = 0; i < TSIZE; i++) {
    if (i < TSIZE - 1 || i * DIM + threadIdx.x < preNmsTopN) {
      t[i] = cProposals[i * DIM + threadIdx.x];
    }
  }

  __shared__ Bbox<T_PROPOSALS> last;
  __shared__ bool kept;
  __shared__ int foundBatch;
  if (threadIdx.x == 0)
    foundBatch = 0;

  for (int i = 0; i < TSIZE; i++) {
    for (int j = 0; j < DIM; j++) {
      int offset = i * DIM;
      int index = offset + j;
      if (index >= preNmsTopN)
        break;

      __syncthreads();

      if (threadIdx.x == j) {
        kept = 0 == (del & ((uint64_t)1 << i));
        last = t[i];

        if (kept) {
          int cnt = blockIdx.x * afterNmsTopN + foundBatch;
          filtered[cnt * 4 + 0] = t[i].xmin;
          filtered[cnt * 4 + 1] = t[i].ymin;
          filtered[cnt * 4 + 2] = t[i].xmax;
          filtered[cnt * 4 + 3] = t[i].ymax;
          foundBatch++;
        }
      }

      __syncthreads();

      if (foundBatch == afterNmsTopN) {
        return;
      }

      if (kept) {
        Bbox<T_PROPOSALS> test = last;

        for (int k = 0; k < TSIZE; k++) {
          if (index < k * DIM + threadIdx.x &&
              IoU<T_PROPOSALS>(test, t[k]) > nmsThres) {
            del |= (uint64_t)1 << k;
          }
        }
      }
    }
  }
}

// NMS LAUNCH
template <typename T_PROPOSALS, DLayout_t L_PROPOSALS, typename T_ROIS>
pluginStatus_t nmsLaunch(
    {{prefix}}Stream_t stream,
    const int batch,
    const int propSize,
    void* proposals,
    void* filtered,
    const int preNmsTopN,
    const float nmsThres,
    const int afterNmsTopN) {
  const int blockSize = 1024;

  // #define P1(tsize) nmsKernel1<T_PROPOSALS, T_ROIS, blockSize, (tsize)>
  // #define P2(tsize) nmsKernel2<T_PROPOSALS, T_ROIS, blockSize, (tsize)>

  //   void (*kernel[64])(
  //       int, Bbox<T_PROPOSALS> const*, T_ROIS*, int, float, int) = {
  //       P1(1),  P1(2),  P1(3),  P1(4),  P1(5),  P1(6),  P1(7),  P1(8),
  //       P1(9),  P1(10), P1(11), P1(12), P2(13), P2(14), P2(15), P2(16),
  //       P2(17), P2(18), P2(19), P2(20), P2(21), P2(22), P2(23), P2(24),
  //       P2(25), P2(26), P2(27), P2(28), P2(29), P2(30), P2(31), P2(32),
  //       P2(33), P2(34), P2(35), P2(36), P2(37), P2(38), P2(39), P2(40),
  //       P2(41), P2(42), P2(43), P2(44), P2(45), P2(46), P2(47), P2(48),
  //       P2(49), P2(50), P2(51), P2(52), P2(53), P2(54), P2(55), P2(56),
  //       P2(57), P2(58), P2(59), P2(60), P2(61), P2(62), P2(63), P2(64)};

#if T_SZIE <= 12
#define nmsKernel nmsKernel1<T_PROPOSALS, T_ROIS, blockSize, T_SIZE>
#else
#define nmsKernel nmsKernel2<T_PROPOSALS, T_ROIS, blockSize, T_SIZE>
#endif

  ASSERT_PARAM(preNmsTopN < 64 * blockSize);

  CSC({{prefix}}MemsetAsync(
          filtered, 0x00, batch * afterNmsTopN * 4 * sizeof(T_ROIS), stream),
      STATUS_FAILURE);

  nmsKernel<<<batch, blockSize, 0, stream>>>(
      propSize,
      (Bbox<T_PROPOSALS>*)proposals,
      (T_ROIS*)filtered,
      preNmsTopN,
      nmsThres,
      afterNmsTopN);

  CSC({{prefix}}GetLastError(), STATUS_FAILURE);

  return STATUS_SUCCESS;
}

// SET OFFSET
// Works for up to 2Gi elements (cub's limitation)!
__global__ void setOffset(int stride, int size, int* output) {
  // One block, because batch size shouldn't be too large.
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    output[i] = i * stride;
  }
}

// BBFilter KERNEL half
__global__ void bboxFilter_kernel(
    int N,
    const float minSize,
    const half* proposals,
    half* scores) {
  if (minSize == 0)
    return;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint16_t bits = 0x3c00u;
  half one = reinterpret_cast<half const&>(bits);

  if (tid < N) {
    int ininf = 0xff800000;
    float ninf = *(float*)&ininf;

    if (__hsub(proposals[tid * 4 + 2], proposals[tid * 4 + 0]) <
            half(minSize) ||
        __hsub(proposals[tid * 4 + 3], proposals[tid * 4 + 1]) <
            half(minSize)) {
      scores[tid] = half(ninf);
    }
  }
}

// BBFilter KERNEL float
__global__ void bboxFilter_kernel(
    int N,
    const float minSize,
    const float* proposals,
    float* scores) {
  if (minSize == 0)
    return;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    int ininf = 0xff800000;
    float ninf = *(float*)&ininf;

    if (proposals[tid * 4 + 2] - proposals[tid * 4 + 0] < minSize ||
        proposals[tid * 4 + 3] - proposals[tid * 4 + 1] < minSize) {
      scores[tid] = ninf;
    }
  }
}

inline size_t GetCudaAlignedSize(size_t size) {
  const size_t kCudaAlignSize = 1 << 20;
  return (size + kCudaAlignSize - 1) / kCudaAlignSize * kCudaAlignSize;
}

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

template <typename KeyType, typename ValueType>
size_t InferTempStorageForSortPairsDescending(
    int32_t num_row,
    int32_t num_col) {
  using SegmentOffsetIter = {{cub}}::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      {{cub}}::CountingInputIterator<int32_t>>;

  {{cub}}::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  size_t temp_storage_bytes = 0;
  auto err = {{cub}}::DeviceSegmentedRadixSort::
      SortPairsDescending<KeyType, ValueType, SegmentOffsetIter>(
          /* d_temp_storage */ nullptr,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_keys_in */ nullptr,
          /* d_keys_out */ nullptr,
          /* d_values_in */ nullptr,
          /* d_values_out */ nullptr,
          /* num_items */ num_row * num_col,
          /* num_segments */ num_row,
          /* d_begin_offsets */ segment_offset_iter,
          /* d_end_offsets */ segment_offset_iter + 1,
          /* begin_bit */ 0,
          /* end_bit */ sizeof(KeyType) * 8,
          /* stream */ 0);
  // OF_CUDA_CHECK(err);

  return temp_storage_bytes;
}

// NMS GPU
template <typename T_SCORES, typename T_ROIS>
pluginStatus_t nmsGpu(
    {{prefix}}Stream_t stream,
    const int N,
    const int R,
    const int preNmsTop,
    const int nmsMaxOut,
    const float iouThreshold,
    const float minBoxSize,
    const void* fgScores,
    const void* proposals,
    void* workspace,
    void* rois) {
  const int BS = 32;
  const int GS = ((R) + BS - 1) / BS;
  bboxFilter_kernel<<<GS, BS, 0, stream>>>(
      R, minBoxSize, (T_ROIS*)proposals, (T_ROIS*)fgScores);

  int8_t* vworkspace = alignPtr((int8_t*)workspace, 32);

  pluginStatus_t error;

  int* offsets = (int*)vworkspace;
  setOffset<<<1, 1024, 0, stream>>>(R, N + 1, offsets);
  CSC({{prefix}}GetLastError(), STATUS_FAILURE);

  vworkspace = vworkspace + N + 1;
  vworkspace = alignPtr(vworkspace, ALIGNMENT);

  std::size_t tempStorageBytes =
      InferTempStorageForSortPairsDescending<T_ROIS, Bbox<T_ROIS>>(N, R);

  CSC({{prefix}}GetLastError(), STATUS_FAILURE);

  T_SCORES* scoresOut = (T_SCORES*)vworkspace;
  vworkspace = (int8_t*)(scoresOut + N * R);
  vworkspace = alignPtr(vworkspace, ALIGNMENT);
  Bbox<T_ROIS>* proposalsOut = (Bbox<T_ROIS>*)vworkspace;
  vworkspace = (int8_t*)(proposalsOut + N * R);
  vworkspace = alignPtr(vworkspace, ALIGNMENT);

  {{cub}}::DeviceSegmentedRadixSort::SortPairsDescending(
      vworkspace,
      tempStorageBytes,
      (T_SCORES*)fgScores,
      (T_SCORES*)scoresOut,
      (Bbox<T_ROIS>*)proposals,
      (Bbox<T_ROIS>*)proposalsOut,
      N * R,
      N,
      offsets,
      offsets + 1,
      0,
      8 * sizeof(T_SCORES),
      stream);

  CSC({{prefix}}GetLastError(), STATUS_FAILURE);

  error = nmsLaunch<T_ROIS, NC4HW, T_ROIS>(
      stream, N, R, proposalsOut, rois, preNmsTop, iouThreshold, nmsMaxOut);

  if (error != STATUS_SUCCESS) {
    return error;
  }
  return STATUS_SUCCESS;
}
    """
)
