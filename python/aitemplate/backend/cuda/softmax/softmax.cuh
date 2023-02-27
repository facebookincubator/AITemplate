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

#ifndef CUDA_SOFTMAX
#define CUDA_SOFTMAX

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cassert>
#include <stdexcept>
#include <string>

using bfloat16 = nv_bfloat16;

#define SOFTMAX_DEVICE_CHECK(call)                                   \
  if ((call) != cudaSuccess) {                                       \
    throw std::runtime_error(                                        \
        std::string("softmax kernel call failed: ") +                \
        cudaGetErrorString(cudaGetLastError()) + " at " + __FILE__ + \
        ", line" + std::to_string(__LINE__));                        \
  }

#define SOFTMAX_LAUNCH_CHECK() SOFTMAX_DEVICE_CHECK(cudaGetLastError())

// unroll directives copied from CUTLASS
#if defined(__CUDA_ARCH__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define PRAGMA_UNROLL _Pragma("unroll")
#else
#define PRAGMA_UNROLL #pragma unroll
#endif // __CUDACC_RTC__

#else
#define PRAGMA_UNROLL
#endif // __CUDA_ARCH__

namespace {

template <typename T>
__inline__ __device__ T fast_max(const T a, const T b);

template <typename T>
__inline__ __device__ T fast_exp(const T a);

template <>
__inline__ __device__ half fast_max(const half a, const half b) {
#if (__CUDA_ARCH__ >= 800)
  return __hmax(a, b);
#else
  return a > b ? a : b;
#endif
}

template <>
__inline__ __device__ float fast_max(const float a, const float b) {
  return fmaxf(a, b);
}

template <>
__inline__ __device__ half fast_exp(const half a) {
  return hexp(a);
}

template <>
__inline__ __device__ float fast_exp(const float a) {
  return __expf(a);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

template <>
__inline__ __device__ bfloat16 fast_exp(const bfloat16 a) {
  return hexp(a);
}

template <>
__inline__ __device__ bfloat16 fast_max(const bfloat16 a, const bfloat16 b) {
  return __hmax(a, b);
}

#endif

template <typename T>
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template <>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template <typename T>
struct Arguments {
  const T* input;
  T* output;
};

struct float8 {
  float4 f0;
  float4 f1;
};

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T* val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T* val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f; // threadIdx.x % warp_size
  int wid = threadIdx.x >> 5; // threadIdx.x / warp_size

  warpReduceSum<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  if (wid == 0)
    warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] = max(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
    }
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceMax(T* val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceMax<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  if (wid == 0)
    warpReduceMax<T, NUM>(val);
  return (T)0.0f;
}

} // namespace

// input size: [M, K]
// Currently the softmax kernel only supports 2D input with dim=1.
// For input with more dimensions, reshape first.
// This kernel is fast for even K, but slow for odd K (K >= 15).
// dtype=float is not tested.

// each thread reduces a tile of size [m, K]
// m is the tile size in M dim
template <
    typename T,
    typename VECTORIZED_TYPE,
    int num_thread,
    size_t K,
    size_t m>
__global__ void softmax_small_k(Arguments<T> args, size_t M) {
  const size_t idx = blockIdx.x * num_thread + threadIdx.x;
  const size_t m_idx = m * idx;

  if (m_idx >= M) {
    return;
  }

  constexpr size_t vector_len = sizeof(VECTORIZED_TYPE) / sizeof(T);
  constexpr bool can_use_vector_load = ((m * K) % vector_len) == 0;
  // read input
  if (can_use_vector_load && m_idx + m < M) {
    auto input = reinterpret_cast<const VECTORIZED_TYPE*>(args.input);
    VECTORIZED_TYPE* output = reinterpret_cast<VECTORIZED_TYPE*>(args.output);

    const size_t offset = (m_idx * K) / vector_len;
    input += offset;
    output += offset;

    static_assert(m <= 8, "tile size m should always be <= 8");

    // round up to make compiler happy
    constexpr int n_tile = (m * K + vector_len - 1) / vector_len;
    VECTORIZED_TYPE input_tile_vec[n_tile];
    T* input_tile = reinterpret_cast<T*>(&input_tile_vec);

    PRAGMA_UNROLL
    for (size_t i = 0; i < n_tile; i++) {
      input_tile_vec[i] = input[i];
    }

    PRAGMA_UNROLL
    for (size_t i = 0; i < m; i++) {
      T max = std::numeric_limits<T>::lowest();
      // find max
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        max = fast_max(input_tile[i * K + j], max);
      }
      // get sum
      float sum = 0;
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        const int tile_idx = i * K + j;
        input_tile[tile_idx] = fast_exp(input_tile[tile_idx] - max);
        sum += static_cast<float>(input_tile[tile_idx]);
      }
      // normalize
      const float sum_inverse = 1.0 / sum;
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        const int tile_idx = i * K + j;
        input_tile[tile_idx] = static_cast<T>(
            static_cast<float>(input_tile[tile_idx]) * sum_inverse);
      }
    }
    PRAGMA_UNROLL
    for (size_t i = 0; i < n_tile; i++) {
      output[i] = input_tile_vec[i];
    }
  } else {
    const T* input = args.input;
    T* output = args.output;

    const size_t offset = m_idx * K;
    input += offset;
    output += offset;

    // handles both odd K and tail batches
    const size_t real_m = M - m_idx >= m ? m : M - m_idx;

    for (size_t i = 0; i < real_m; i++) {
      T input_tile[K];

      // read input
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        input_tile[j] = input[i * K + j];
      }

      T max = std::numeric_limits<T>::lowest();
      // find max
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        max = fast_max(input_tile[j], max);
      }
      // get sum
      float sum = 0;
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        const int tile_idx = i * K + j;
        input_tile[j] = fast_exp(input_tile[j] - max);
        sum += static_cast<float>(input_tile[j]);
      }
      // normalize
      float sum_inverse = 1.0 / sum;
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        input_tile[j] =
            static_cast<T>(static_cast<float>(input_tile[j]) * sum_inverse);
      }
      // write output
      PRAGMA_UNROLL
      for (size_t j = 0; j < K; j++) {
        output[i * K + j] = input_tile[j];
      }
    }
  }
}

// This is a special case where K is really large, we still use block reduction.
// In this case, we won’t have enough shared memory and we will not cache any
// kernel. i.e. we no longer keep shared memory, but calculate exp(buf[i]-s_max)
// each time we need it.
template <typename T>
__global__ void softmaxBlockNocache(
    T* input,
    T* output,
    size_t m,
    const size_t n) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_max, s_sum;
  int offset = m_idx * n;
  input += offset;
  output += offset;

  float local_max[1] = {-Inf<float>()};
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(input[i]);
    local_max[0] = max(local_val, local_max[0]);
  }

  if (blockDim.x <= 32) {
    warpReduceMax<float, 1>(local_max);
  } else {
    blockReduceMax<float, 1>(local_max);
  }
  if (threadIdx.x == 0) {
    s_max = local_max[0];
  }
  __syncthreads();
  float local_sum[1] = {0.0f};
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum[0] += exp(static_cast<float>(input[i]) - s_max);
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sum);
  } else {
    blockReduceSum<float, 1>(local_sum);
  }
  if (threadIdx.x == 0) {
    s_sum = local_sum[0];
  }
  __syncthreads();
  for (int i = tid; i < n; i += blockDim.x) {
    output[i] = T(exp(static_cast<float>(input[i]) - s_max) / s_sum);
  }
}

// Assuming input[M, K], we use vector read with pack_size as length.
// There are two cases:
// 1) When K/pack_size >= 32.* We launch M/pack_size blocks and 128 threads.
// Each block is further partition into two dimensions x and y,
// where on x dimension we perform wrap reduction on columns, on y dimension we
// parallelize independent row operations. The warp size is 32 as K >
// 32*pack_size. i.e. GridDim = <M/pack_size>, BlockDim = <32, 4>. Each thread
// processes K/32 columns. Each block processes 4 rows, 32 columns. Each grid
// processes M/4 rows. 2) When K/pack_size < 32.* We launch M*K/pack_size/128
// blocks and 128 threads. Each block is further partition into two dimensions x
// and y, where on x dimension we perform wrap reduction on columns, on y
// dimension we parallelize independent row operations. But this time the wrap
// size is K/pack_size i.e. GridDim = <MK/128/pack_size>, BlockDim =
// <K/pack_size, 128/K*pack_size> Each thread processes pack_size columns.
// (pack_size) Each block processes 128/K*pack_size rows, K/pack_size columns.
// Each grid processes M*K/128/pack_size rows.

template <typename T, typename ACT_T, int cols_per_thread>
__global__ void softmax_stored_locally_multi_dim(
    const T* input,
    T* output,
    size_t m,
    size_t n) {
  const int read_t_sz = sizeof(T);
  const int act_t_sz = sizeof(ACT_T);
  const int pack_size = read_t_sz / act_t_sz;

  constexpr int num_packs = (cols_per_thread + pack_size - 1) / pack_size;
  float buf[cols_per_thread];
  const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {
    const int64_t row_offset = row * int((n + pack_size - 1) / pack_size);
    const T* row_x = input + row_offset;
    T* row_y = output + row_offset;
    float local_max[1] = {-Inf<float>()};
#pragma unroll
    for (int i = 0; i < num_packs; ++i) {
      const int col = i * blockDim.x + tid;
      T tmp_in = row_x[col];
      const ACT_T* pack_x = reinterpret_cast<const ACT_T*>(&tmp_in);
      if (col < n / pack_size) {
#pragma unroll
        for (int j = 0; j < pack_size; j++) {
          buf[i * pack_size + j] = static_cast<float>(pack_x[j]);
          local_max[0] = max(local_max[0], buf[i * pack_size + j]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < pack_size; j++) {
          buf[i * pack_size + j] = -Inf<float>();
        }
      }
    }
    warpReduceMax<float, 1>(local_max, blockDim.x);

    float local_sum[1] = {0.0f};
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
      buf[i] = exp(buf[i] - local_max[0]);
      local_sum[0] += buf[i];
    }
    warpReduceSum<float, 1>(local_sum, blockDim.x);

    T tmp_o;
    ACT_T* pack_y = reinterpret_cast<ACT_T*>(&tmp_o);
#pragma unroll
    for (int i = 0; i < num_packs; i++) {
      const int col = i * blockDim.x + tid;
      if (col < n / pack_size) {
        for (int j = 0; j < pack_size; j++) {
          pack_y[j] = ACT_T(buf[i * pack_size + j] / local_sum[0]);
        }
        row_y[col] = tmp_o;
      }
    }
  }
}

template <typename T, typename ACT_T, int block_size>
__global__ void softmax_block_smem(
    const T* input,
    T* output,
    size_t m,
    const size_t n) {
  const int read_t_sz = sizeof(T);
  const int act_t_sz = sizeof(ACT_T);
  const int pack_size = read_t_sz / act_t_sz;

  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ __align__(sizeof(
      float)) unsigned char shared_buf[]; // size_t smem = n*sizeof(float)
  auto* buf = reinterpret_cast<float*>(shared_buf);
  const int num_packs = (n + pack_size - 1) / pack_size;
  for (int64_t row = m_idx; row < m; row += gridDim.x) {
    const int64_t row_offset = row * int((n + pack_size - 1) / pack_size);
    const T* row_x = input + row_offset;
    T* row_y = output + row_offset;
    float local_max[1] = {-Inf<float>()};

    for (int pack_id = tid; pack_id < num_packs; pack_id += blockDim.x) {
      T tmp_in = row_x[pack_id];
      const ACT_T* pack_x = reinterpret_cast<const ACT_T*>(&tmp_in);
      // store to local register, which is faster than shared memory
      for (int j = 0; j < pack_size; j++) {
        float pack = pack_x[j];
        buf[j * num_packs + pack_id] = pack;
        local_max[0] = max(local_max[0], pack);
      }
    }
    blockReduceMax<float, 1>(local_max); // reduce on a block of #blockDim.x

    __shared__ float s_max;
    if (threadIdx.x == 0) {
      s_max = local_max[0];
    }
    __syncthreads();

    float local_sum[1] = {0.0f};
    for (int i = tid; i < n; i += blockDim.x) {
      float local_val = exp(buf[i] - s_max);
      buf[i] = local_val;
      local_sum[0] += local_val;
    }
    blockReduceSum<float, 1>(local_sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) {
      s_sum = local_sum[0];
    }
    __syncthreads();

    T tmp_o;
    ACT_T* pack_y = reinterpret_cast<ACT_T*>(&tmp_o);

    for (int i = tid; i < num_packs; i += blockDim.x) {
      for (int j = 0; j < pack_size; j++) {
        const int col = i + j * num_packs;
        pack_y[j] = ACT_T(buf[col] / s_sum);
      }
      row_y[i] = tmp_o;
    }
  }
}

// We launch M blocks and 1024 (maximum) threads. Each block handles a column
// and we launch as many blocks as #rows. i.e. We launch GridDim = <M>, BlockDim
// = <block_size>, Shared memory = K*sizeof(float). The block_size can be one of
// 1024, 512, 256, 128. We first use
// cudaOccupancyMaxActiveBlocksPerMultiprocessor to calculate actual used
// threads. If there is no waste, we would like it to be as large as possible to
// achieve higher concurrency (e.g 1024). Each thread processes K/block_size
// columns. Each block processes block_size columns. Each grid processes M rows.
template <typename T, typename ACT_T, size_t n>
inline cudaError_t LaunchSoftmaxBlockAll(
    const T* input,
    T* output,
    size_t m,
    cudaStream_t stream,
    bool* success) {
  unsigned read_t_sz = sizeof(T);
  unsigned comp_t_sz = sizeof(ACT_T);
  unsigned pack_size = read_t_sz / comp_t_sz;
  dim3 grid(m);
  dim3 block(int((n + pack_size - 1) / pack_size));
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = n * sizeof(float);
  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        softmax_block_smem<T, ACT_T, block_size_conf_1>,
        block_size_conf_1,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  *success = true;
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        softmax_block_smem<T, ACT_T, block_size_conf_4>,
        block_size_conf_4,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    softmax_block_smem<T, ACT_T, block_size_conf_4>
        <<<grid, block_size_conf_4, smem, stream>>>(input, output, m, n);
    return cudaSuccess;
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        softmax_block_smem<T, ACT_T, block_size_conf_3>,
        block_size_conf_3,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    softmax_block_smem<T, ACT_T, block_size_conf_3>
        <<<grid, block_size_conf_3, smem, stream>>>(input, output, m, n);
    return cudaSuccess;
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        softmax_block_smem<T, ACT_T, block_size_conf_2>,
        block_size_conf_2,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    softmax_block_smem<T, ACT_T, block_size_conf_2>
        <<<grid, block_size_conf_2, smem, stream>>>(input, output, m, n);
    return cudaSuccess;
  }
  softmax_block_smem<T, ACT_T, block_size_conf_1>
      <<<grid, block_size_conf_1, smem, stream>>>(input, output, m, n);
  return cudaSuccess;
}

template <typename T, int K, size_t TileSize>
void LaunchSoftmaxSmallK(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  const int n_threads = 128;
  const int tile_size_by_n_threads = TileSize * n_threads;
  dim3 block(n_threads);
  dim3 grid((batch_size + tile_size_by_n_threads - 1) / tile_size_by_n_threads);
  softmax_small_k<T, float4, n_threads, K, TileSize>
      <<<grid, block, 0, stream>>>({input, output}, batch_size);
  SOFTMAX_LAUNCH_CHECK();
}

template <typename T>
struct VecTFor;

template <>
struct VecTFor<half> {
  using vec8 = float4;
  using vec4 = float2;
  using vec2 = float;
};

template <>
struct VecTFor<float> {
  using vec8 = float8;
  using vec4 = float4;
  using vec2 = float2;
};

template <>
struct VecTFor<bfloat16> {
  using vec8 = float4;
  using vec4 = float2;
  using vec2 = float;
};

template <typename T, size_t NElements>
void LaunchSoftmaxK8Small(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_width = -1;
  for (auto i : {1, 8, 16, 32}) {
    if (8 * i >= NElements) {
      thread_group_width = i;
      break;
    }
  }
  int thread_group_per_block = 128 / thread_group_width;
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(thread_group_width, thread_group_per_block);
  using vec8 = typename VecTFor<T>::vec8;
  softmax_stored_locally_multi_dim<vec8, T, 8><<<grid, block, 0, stream>>>(
      reinterpret_cast<const vec8*>(input),
      reinterpret_cast<vec8*>(output),
      batch_size,
      NElements);
  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK8Middle(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_per_block = 128 / 32; // 4
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(32, thread_group_per_block);
  const int num_packs = (int((NElements + 31) / 32) + 7) / 8;
  const int cols_per_thread = num_packs * 8;
  using vec8 = typename VecTFor<T>::vec8;
  softmax_stored_locally_multi_dim<vec8, T, cols_per_thread>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const vec8*>(input),
          reinterpret_cast<vec8*>(output),
          batch_size,
          NElements);
  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK4Small(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_width = -1;
  for (auto i : {1, 4, 8, 16, 32}) {
    if (4 * i >= NElements) {
      thread_group_width = i;
      break;
    }
  }
  int thread_group_per_block = 128 / thread_group_width;
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(thread_group_width, thread_group_per_block);
  using vec4 = typename VecTFor<T>::vec4;
  softmax_stored_locally_multi_dim<vec4, T, 8><<<grid, block, 0, stream>>>(
      reinterpret_cast<const vec4*>(input),
      reinterpret_cast<vec4*>(output),
      batch_size,
      NElements);
  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK4Middle(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_per_block = 128 / 32; // 4
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(32, thread_group_per_block);
  const int num_packs = (int((NElements + 31) / 32) + 3) / 4;
  const int cols_per_thread = num_packs * 8;
  using vec4 = typename VecTFor<T>::vec4;

  softmax_stored_locally_multi_dim<vec4, T, cols_per_thread>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const vec4*>(input),
          reinterpret_cast<vec4*>(output),
          batch_size,
          NElements);

  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK2Small(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_width = -1;
  for (auto i : {1, 2, 4, 8, 16, 32}) {
    if (2 * i >= NElements) {
      thread_group_width = i;
      break;
    }
  }
  int thread_group_per_block = 128 / thread_group_width;
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(thread_group_width, thread_group_per_block);
  using vec2 = typename VecTFor<T>::vec2;

  softmax_stored_locally_multi_dim<vec2, T, 8><<<grid, block, 0, stream>>>(
      reinterpret_cast<const vec2*>(input),
      reinterpret_cast<vec2*>(output),
      batch_size,
      NElements);

  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK2Middle(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_per_block = 128 / 32; // 4
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(32, thread_group_per_block);
  const int num_packs = (int((NElements + 31) / 32) + 1) / 2;
  const int cols_per_thread = num_packs * 2;
  using vec2 = typename VecTFor<T>::vec2;

  softmax_stored_locally_multi_dim<vec2, T, cols_per_thread>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const vec2*>(input),
          reinterpret_cast<vec2*>(output),
          batch_size,
          NElements);

  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK1Small(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_width = -1;
  for (auto i : {1, 2, 4, 8, 16, 32}) {
    if (i >= NElements) {
      thread_group_width = i;
      break;
    }
  }
  int thread_group_per_block = 128 / thread_group_width;
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(thread_group_width, thread_group_per_block);

  softmax_stored_locally_multi_dim<T, T, 8>
      <<<grid, block, 0, stream>>>(input, output, batch_size, NElements);

  SOFTMAX_LAUNCH_CHECK();
}

template <typename T, size_t NElements>
void LaunchSoftmaxK1Middle(
    const T* input,
    T* output,
    size_t batch_size,
    cudaStream_t stream) {
  int thread_group_per_block = 128 / 32; // 4
  int grid_dim_x =
      (batch_size + thread_group_per_block - 1) / thread_group_per_block;
  dim3 grid(grid_dim_x);
  dim3 block(32, thread_group_per_block);
  const int cols_per_thread = (NElements + 31) / 32;

  softmax_stored_locally_multi_dim<T, T, cols_per_thread>
      <<<grid, block, 0, stream>>>(input, output, batch_size, NElements);

  SOFTMAX_LAUNCH_CHECK();
}

#endif
