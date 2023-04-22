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
#ifndef GROUPNORM_KERNEL_CUH
#define GROUPNORM_KERNEL_CUH

constexpr uint32_t kFinalMask = 0xffffffff;

#ifndef GROUP_NORM_CUDA_CHECK
#define GROUP_NORM_CUDA_CHECK(expr)                                       \
  do {                                                                    \
    cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " \
                << __FILE__ << ": " << __LINE__ << std::endl;             \
      return status;                                                      \
    }                                                                     \
  } while (0)
#endif

#ifndef GROUP_NORM_CUDA_CHECK_LAUNCH
#define GROUP_NORM_CUDA_CHECK_LAUNCH() GROUP_NORM_CUDA_CHECK(cudaGetLastError())
#endif

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#endif

#define NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)

__device__ half fast_tanh(half x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 750)

  asm volatile("tanh.approx.f16 %0, %1;"
               : "=h"(__HALF_TO_US(x))
               : "h"(__HALF_TO_US(x)));
  return x;

#else
  return half(cutlass::fast_tanh(float(x)));
#endif
}

__device__ bfloat16 fast_tanh(bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 900)
  asm volatile("tanh.approx.bf16 %0, %1;"
               : "=h"(__HALF_TO_US(x))
               : "h"(__HALF_TO_US(x)));
  return x;

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return cutlass::fast_tanh(float(x));
#else
  NOT_IMPLEMENTED();
#endif
}

#define CUDA_FP16_ONE_HALF \
  __half_raw {             \
    0x3800u                \
  }
#define CUDA_FP16_ONE \
  __half_raw {        \
    0x3c00u           \
  }
#define CUDA_BF16_ONE_HALF \
  __nv_bfloat16_raw {      \
    0x3f00u                \
  }
#define CUDA_BF16_ONE \
  __nv_bfloat16_raw { \
    0x3f80u           \
  }

__device__ float sigmoid(const float a) {
  return (cutlass::fast_tanh(a * 0.5f) + 1.0f) * 0.5f;
}

__device__ half hsigmoid(const half a) {
  return __hmul(
      (__hadd(fast_tanh(__hmul(a, CUDA_FP16_ONE_HALF)), CUDA_FP16_ONE)),
      CUDA_FP16_ONE_HALF);
}

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 800)
__device__ bfloat16 bf16sigmoid(const bfloat16 a) {
  return __hmul(
      (__hadd(fast_tanh(__hmul(a, CUDA_BF16_ONE_HALF)), CUDA_BF16_ONE)),
      CUDA_BF16_ONE_HALF);
}
#endif

template <typename T>
struct FSigmoid {
  __inline__ __device__ T operator()(const T input) const;
};

template <>
struct FSigmoid<half> {
  __inline__ __device__ half operator()(const half a) const {
    return hsigmoid(a);
  }
};

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 800)
template <>
struct FSigmoid<bfloat16> {
  __inline__ __device__ bfloat16 operator()(const bfloat16 a) const {
    return bf16sigmoid(a);
  }
};
#endif

template <>
struct FSigmoid<float> {
  __inline__ __device__ float operator()(const float a) const {
    return sigmoid(a);
  }
};

////////////////////////////////////////////////////////////////////////////////
// The Groupnorm implementation below is based on OneFlow's Layernorm
// implementation at:
// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh

/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#define __AIT_GN_USE_FAST_MATH 1
template <typename T>
__forceinline__ __device__ T Div(T a, T b);

template <>
__forceinline__ __device__ float Div<float>(float a, float b) {
#ifdef __AIT_GN_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template <>
__forceinline__ __device__ half Div<half>(half a, half b) {
  return __hdiv(a, b);
}

template <typename T>
__forceinline__ __device__ T Rsqrt(T x);

template <>
__forceinline__ __device__ float Rsqrt<float>(float x) {
#ifdef __AIT_GN_USE_FAST_MATH
  return __frsqrt_rn(x);
#else
  return rsqrt(x);
#endif
}

template <>
__forceinline__ __device__ half Rsqrt<half>(half x) {
  return hrsqrt(x);
}

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 800)
template <>
__forceinline__ __device__ bfloat16 Rsqrt<bfloat16>(bfloat16 x) {
  return hrsqrt(x);
}
#endif

#undef __AIT_GN_USE_FAST_MATH

template <typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, int* count) {
  // Use Welford Online algorithm to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, static_cast<T>(*count));
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

template <typename T>
inline __device__ void WelfordCombine(
    T b_mean,
    T b_m2,
    int b_count,
    T* mean,
    T* m2,
    int* count) {
  if (b_count == 0) {
    return;
  }
  int new_count = *count + b_count;
  T nb_over_n = Div((T)b_count, (T)new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (T)(*count) * (T)(nb_over_n);
  *count = new_count;
}

constexpr int kWarpSize = 32;

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(
    T thread_mean,
    T thread_m2,
    int thread_count,
    T* mean,
    T* m2,
    int* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(kFinalMask, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(kFinalMask, *m2, mask, thread_group_width);
    int b_count =
        __shfl_down_sync(kFinalMask, *count, mask, thread_group_width);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template <typename T>
__inline__ __device__ void WelfordBlockAllReduce(
    T thread_mean,
    T thread_m2,
    int thread_count,
    T* result_mean,
    T* result_m2,
    int* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ int count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ int count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  int warp_count = 0;
  WelfordWarpReduce(
      thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    int block_count = 0;
    WelfordWarpReduce(
        warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}

template <typename T, typename ComputeType, bool FuseSwish>
__global__ void groupnorm_welford_fp16(
    T* output,
    T* input,
    T* gamma,
    T* beta,
    const float eps,
    const int64_t elems_per_block,
    const int64_t elems_per_group_channel,
    const int64_t batch_stride,
    const int64_t group_stride,
    const int64_t num_rows,
    const int64_t row_stride) {
  // all the numbers and strides are counted with respect to type T
  constexpr int vec_size = sizeof(T) / sizeof(half);

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int gid = blockIdx.y; // index of group
  const int64_t batch_offset = bid * batch_stride;
  const int64_t group_offset = gid * group_stride;
  const int64_t offset = batch_offset + group_offset;

  // the first input of this thread
  const T* t_input = input + offset;

  ComputeType thread_mean = ComputeType(0.0);
  ComputeType thread_m2 = ComputeType(0.0);
  int thread_count = 0;
#pragma unroll
  for (int row_id = tid; row_id < num_rows; row_id += blockDim.x) {
#pragma unroll
    for (int i = 0; i < elems_per_group_channel; i++) {
      const T* local_input = t_input + i + row_id * row_stride;
      const half* half_ptr = reinterpret_cast<const half*>(local_input);
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        WelfordCombine(
            __half2float(half_ptr[j]), &thread_mean, &thread_m2, &thread_count);
      }
    }
  }
  ComputeType row_mean = (ComputeType)(0.0f);
  ComputeType row_m2 = (ComputeType)(0.0f);
  int row_count = 0;
  if (blockDim.x <= 32) {
    WelfordWarpReduce(
        thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);
  } else {
    WelfordBlockAllReduce<ComputeType>(
        thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);
  }
  ComputeType row_variance = Div(row_m2, static_cast<ComputeType>(row_count));
  ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(eps));

  float local_row_mean;
  if (std::is_same<ComputeType, half>::value) {
    local_row_mean = __half2float(row_mean);
  } else if (std::is_same<ComputeType, float>::value) {
    local_row_mean = row_mean;
  }
  float local_row_inv_var;
  if (std::is_same<ComputeType, half>::value) {
    local_row_inv_var = __half2float(row_inv_var);
  } else if (std::is_same<ComputeType, float>::value) {
    local_row_inv_var = row_inv_var;
  }

  const T* t_gamma = gamma + group_offset;
  const T* t_beta = beta + group_offset;
  // the first input of this thread
  T* t_output = output + offset;
#pragma unroll
  for (int row_id = tid; row_id < num_rows; row_id += blockDim.x) {
#pragma unroll
    for (int i = 0; i < elems_per_group_channel; i++) {
      const T* local_input = t_input + i + row_id * row_stride;
      const half* input_half_ptr = reinterpret_cast<const half*>(local_input);

      T* local_output = t_output + i + row_id * row_stride;
      T tmp_output;
      half* output_half_ptr = reinterpret_cast<half*>(&tmp_output);

      const T* local_gamma = t_gamma + i;
      const T* local_beta = t_beta + i;
      const half* gamma_half_ptr = reinterpret_cast<const half*>(local_gamma);
      const half* beta_half_ptr = reinterpret_cast<const half*>(local_beta);

#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        float local_val = __half2float(input_half_ptr[j]);
        float local_gamma = __half2float(gamma_half_ptr[j]);
        float local_beta = __half2float(beta_half_ptr[j]);
        float out_val = (local_val - local_row_mean) * local_row_inv_var;
        out_val = out_val * local_gamma + local_beta;
        out_val = FuseSwish ? out_val * sigmoid(out_val) : out_val;
        output_half_ptr[j] = __float2half_rn(out_val);
      }
      *local_output = tmp_output;
    }
  }
}

// End the Groupnorm implementation that is based on from OneFlow's Layernorm
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) {
    result_broadcast = result;
  }
  __syncthreads();
  return result_broadcast;
}

namespace detail {

template <typename TInput>
struct TInputHelper;

template <>
struct TInputHelper<half> {
  typedef __half2 vec2_type;
  static __inline__ __device__ float2 to_float2(vec2_type a) {
    return __half22float2(a);
  }
  static __inline__ __device__ vec2_type to_vec2(float2 a) {
    return __float22half2_rn(a);
  }
};

template <>
struct TInputHelper<float> {
  typedef float2 vec2_type;
  static __inline__ __device__ float2 to_float2(vec2_type a) {
    return a;
  }
  static __inline__ __device__ vec2_type to_vec2(float2 a) {
    return a;
  }
};

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 800)
template <>
struct TInputHelper<bfloat16> {
  typedef bfloat16_2 vec2_type;
  static __inline__ __device__ float2 to_float2(vec2_type a) {
    return __bfloat1622float2(a);
  }
  static __inline__ __device__ vec2_type to_vec2(float2 a) {
    return __float22bfloat162_rn(a);
  }
};
#endif

} // namespace detail

template <
    typename TInput,
    bool FuseSwish,
    int H,
    int W,
    int C,
    int C_G,
    int ILP = 8,
    int BANK_CONFLICT = 0,
    int NUM_THREADS = 1024>
__global__ __launch_bounds__(NUM_THREADS) void group_norm_smem(
    const TInput* X,
    TInput* Y,
    TInput* gamma,
    TInput* beta,
    int N,
    float epsilon) {
  constexpr int C_G_2 = C_G / 2;
  constexpr int C_G_stride = C_G_2 + BANK_CONFLICT;
  extern __shared__ int svals_[];
  using vec2_type = typename detail::TInputHelper<TInput>::vec2_type;
  auto to_float2 = detail::TInputHelper<TInput>::to_float2;
  auto to_vec2 = detail::TInputHelper<TInput>::to_vec2;
  auto* svals = reinterpret_cast<vec2_type*>(&svals_[0]);

  const int32_t g = blockIdx.x;
  const int32_t start_c = g * C_G;
  const int32_t n = blockIdx.y;

  // X: [N, H, W, C]
  // last stride is 1
  const int32_t src_strides[3] = {H * W * C, W * C, C};
  const int32_t smem_strides[2] = {W * C_G_stride, C_G_stride};
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    const auto c_g_2 = load_idx % C_G_2;
    const auto w = (load_idx / C_G_2) % W;
    const auto h_ilp = ((load_idx / C_G_2) / W);

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      const vec2_type* const src = reinterpret_cast<const vec2_type*>(
          &(X[n * src_strides[0] + (h_ilp * ILP + ii) * src_strides[1] +
              w * src_strides[2] + (start_c + c_g_2 * 2)]));
      vec2_type* const dst = &svals
                                 [(h_ilp * ILP + ii) * smem_strides[0] +
                                  w * smem_strides[1] + c_g_2];
      cutlass::arch::cp_async_zfill<sizeof(vec2_type)>(dst, src, true);
    }
  }
  cutlass::arch::cp_async_wait<0>();

  float thread_sum = 0;
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    const auto c_g_2 = load_idx % C_G_2;
    const auto w = (load_idx / C_G_2) % W;
    const auto h_ilp = ((load_idx / C_G_2) / W);
#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      const vec2_type valh = svals
          [(h_ilp * ILP + ii) * smem_strides[0] + w * smem_strides[1] + c_g_2];
      const float2 val = to_float2(valh);
      thread_sum += val.x + val.y;
    }
  }
  const float block_mean =
      BlockAllReduce<SumOp, float, NUM_THREADS>(thread_sum) /
      float(H * W * C_G);

  float thread_sq_sum = 0;
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    const auto c_g_2 = load_idx % C_G_2;
    const auto w = (load_idx / C_G_2) % W;
    const auto h_ilp = ((load_idx / C_G_2) / W);

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      const vec2_type valh = svals
          [(h_ilp * ILP + ii) * smem_strides[0] + w * smem_strides[1] + c_g_2];
      const float2 val = to_float2(valh);
      thread_sq_sum += (val.x - block_mean) * (val.x - block_mean) +
          (val.y - block_mean) * (val.y - block_mean);
    }
  }
  // PyTorch uses biased estimate of std-dev.
  const float block_inv_std = __frsqrt_rn(
      BlockAllReduce<SumOp, float, NUM_THREADS>(thread_sq_sum) /
          float(H * W * C_G) +
      epsilon);

  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    const auto c_g_2 = load_idx % C_G_2;
    const auto w = (load_idx / C_G_2) % W;
    const auto h_ilp = ((load_idx / C_G_2) / W);

    const auto dst_stride3_offset = start_c + c_g_2 * 2;
    const auto g_v2 =
        *reinterpret_cast<const vec2_type*>(gamma + dst_stride3_offset);
    auto g_f2 = to_float2(g_v2);
    g_f2.x *= block_inv_std;
    g_f2.y *= block_inv_std;
    const auto b_v2 =
        *reinterpret_cast<const vec2_type*>(beta + dst_stride3_offset);
    const auto b_f2 = to_float2(b_v2);

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      const vec2_type src = svals
          [(h_ilp * ILP + ii) * smem_strides[0] + w * smem_strides[1] + c_g_2];
      vec2_type* const dst = reinterpret_cast<vec2_type*>(
          &(Y[n * src_strides[0] + (h_ilp * ILP + ii) * src_strides[1] +
              w * src_strides[2] + dst_stride3_offset]));

      const auto fsrc = to_float2(src);
      float2 result;
      result.x = (fsrc.x - block_mean) * g_f2.x + b_f2.x;
      result.y = (fsrc.y - block_mean) * g_f2.y + b_f2.y;
      if (FuseSwish) {
        result.x = result.x * sigmoid(result.x);
        result.y = result.y * sigmoid(result.y);
      }
      *dst = to_vec2(result);
    }
  }
}

template <bool FuseSwish, int H, int W, int C, int num_groups>
cudaError_t invokeWelfordGroupNorm_half(
    half* output,
    half* input,
    half* gamma,
    half* beta,
    int N,
    const float eps,
    cudaStream_t stream) {
  int max_vec_size = 8;
  while ((C / num_groups) % max_vec_size != 0) {
    max_vec_size /= 2;
  }

  constexpr int64_t block_size = 1024;
  // counts w.r.t. type half
  const int64_t elems_per_group_channel = C / num_groups;
  const int64_t elems_per_block = (H * W * C) / num_groups;
  const int64_t batch_stride = H * W * C;
  const int64_t group_stride = elems_per_group_channel;

  CHECK_EQ(elems_per_group_channel % max_vec_size, 0);
  CHECK_EQ(batch_stride % max_vec_size, 0);
  CHECK_EQ(group_stride % max_vec_size, 0);
  const int64_t v_elems_per_group_channel =
      elems_per_group_channel / max_vec_size;
  const int64_t v_elems_per_block = elems_per_block / max_vec_size;
  const int64_t v_batch_stride = batch_stride / max_vec_size;
  const int64_t v_group_stride = group_stride / max_vec_size;
  const int64_t v_num_rows = v_elems_per_block / v_elems_per_group_channel;
  const int64_t v_row_stride = C / max_vec_size;

  dim3 grid(N, num_groups);

#define __HANDLE_ONE_VEC(vec_type, vec_size)           \
  case vec_size: {                                     \
    groupnorm_welford_fp16<vec_type, float, FuseSwish> \
        <<<grid, block_size, 0, stream>>>(             \
            reinterpret_cast<vec_type*>(output),       \
            reinterpret_cast<vec_type*>(input),        \
            reinterpret_cast<vec_type*>(gamma),        \
            reinterpret_cast<vec_type*>(beta),         \
            eps,                                       \
            v_elems_per_block,                         \
            v_elems_per_group_channel,                 \
            v_batch_stride,                            \
            v_group_stride,                            \
            v_num_rows,                                \
            v_row_stride);                             \
    GROUP_NORM_CUDA_CHECK_LAUNCH();                    \
    break;                                             \
  }

  switch (max_vec_size) {
    __HANDLE_ONE_VEC(uint4, 8)
    __HANDLE_ONE_VEC(uint2, 4)
    __HANDLE_ONE_VEC(unsigned, 2)
    __HANDLE_ONE_VEC(half, 1)
    default:
      throw std::runtime_error("Invalid max_vec_size\n");
  }

#undef __HANDLE_ONE_VEC
  return cudaSuccess;
}

template <typename SRC, typename DST, bool affine, bool FuseSwish>
struct AffineStore {
  AffineStore(
      DST* y,
      int64_t row_size,
      int64_t channel_size,
      int64_t spatial_size,
      const DST* gamma,
      const DST* beta)
      : y(y),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size),
        gamma(gamma),
        beta(beta) {}

  template <int PackSize>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    layer_norm::Pack<DST, PackSize> y_pack;
    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_beta_offset = (offset / spatial_size) % channel_size;
    DST gamma_val = 1.0;
    DST beta_val = 0.0;
    if (affine) {
      gamma_val = gamma[gamma_beta_offset];
      beta_val = beta[gamma_beta_offset];
    }
    FSigmoid<DST> fsigmoid;
#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = normalized_i * gamma_val + beta_val;
      } else {
        // Direct Store.
        y_pack.elem[i] = normalized_i;
      }
      if (FuseSwish) {
        y_pack.elem[i] = y_pack.elem[i] * fsigmoid(y_pack.elem[i]);
      }
    }
    *(reinterpret_cast<layer_norm::PackType<DST, PackSize>*>(y) +
      packed_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) {
    return (spatial_size % pack_size) == 0;
  }
  DST* y;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
  const DST* gamma;
  const DST* beta;
};

template <typename SRC, typename DST, bool affine>
struct ScaleLoad {
  ScaleLoad(
      const SRC* src,
      const SRC* gamma,
      int64_t row_size,
      int64_t channel_size,
      int64_t spatial_size)
      : src(src),
        gamma(gamma),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size) {}
  template <int PackSize>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    layer_norm::Pack<SRC, PackSize> src_pack;
    layer_norm::Pack<SRC, PackSize> gamma_pack;

    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_offset = (offset / spatial_size) % channel_size;

    src_pack.storage =
        *(reinterpret_cast<const layer_norm::PackType<SRC, PackSize>*>(src) +
          packed_offset);
    SRC gamma_val = static_cast<SRC>(1.0);
    if (affine) {
      gamma_val = gamma[gamma_offset];
    }
#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      dst[i] = static_cast<DST>(src_pack.elem[i] * gamma_val);
    }
  }
  bool CanPackAs(size_t pack_size) {
    return (spatial_size % pack_size) == 0;
  }
  const SRC* src;
  const SRC* gamma;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
};

template <typename SRC, typename DST, bool affine, bool FuseSwish>
struct ChannelsLastStore {
  ChannelsLastStore(
      DST* y,
      const DST* gamma,
      const DST* beta,
      int64_t spatial_size,
      int64_t channel_size,
      int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups) {}

  template <int PackSize>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    layer_norm::Pack<DST, PackSize> y_pack;
    layer_norm::Pack<DST, PackSize> gamma_pack;
    layer_norm::Pack<DST, PackSize> beta_pack;
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    const int32_t y_offset =
        (batch_idx * c0.divisor * c1.divisor * spatial_size +
         spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) /
        PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1.divisor + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage = *(
          reinterpret_cast<const layer_norm::PackType<DST, PackSize>*>(gamma) +
          gamma_beta_offset);
      beta_pack.storage =
          *(reinterpret_cast<const layer_norm::PackType<DST, PackSize>*>(beta) +
            gamma_beta_offset);
    }

    FSigmoid<DST> fsigmoid;
#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        // Direct Store.
        y_pack.elem[i] = normalized_i;
      }
      if (FuseSwish) {
        y_pack.elem[i] = y_pack.elem[i] * fsigmoid(y_pack.elem[i]);
      }
    }
    *(reinterpret_cast<layer_norm::PackType<DST, PackSize>*>(y) + y_offset) =
        y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) {
    return (c1.divisor % pack_size) == 0;
  }
  DST* y;
  const DST* gamma;
  const DST* beta;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

template <typename SRC, typename DST>
struct ChannelsLastLoad {
  ChannelsLastLoad(
      const SRC* src,
      int64_t spatial_size,
      int64_t channel_size,
      int64_t num_groups)
      : src(src),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups) {}
  template <int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    layer_norm::Pack<SRC, N> pack;
    const int32_t offset =
        (batch_idx * c0.divisor * c1.divisor * spatial_size +
         spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) /
        N;

    pack.storage =
        *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
  bool CanPackAs(size_t pack_size) {
    return (c1.divisor % pack_size) == 0;
  }
  const SRC* src;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

template <typename T, typename ComputeType, bool affine, bool FuseSwish>
void GroupNormForwardGpu(
    cudaStream_t stream,
    const int64_t num_instances,
    const int64_t norm_size,
    const int64_t channel_size,
    const int64_t spatial_size,
    const double epsilon,
    const T* x_ptr,
    const T* gamma_ptr,
    const T* beta_ptr,
    T* y_ptr,
    ComputeType* mean,
    ComputeType* inv_variance,
    bool channels_first) {
  if (channels_first) {
    layer_norm::DirectLoad<T, ComputeType> load(x_ptr, norm_size);
    AffineStore<ComputeType, T, affine, FuseSwish> store(
        y_ptr, norm_size, channel_size, spatial_size, gamma_ptr, beta_ptr);

    layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
        stream,
        load,
        store,
        num_instances,
        norm_size,
        epsilon,
        mean,
        inv_variance);
  } else {
    ChannelsLastLoad<T, ComputeType> load(
        x_ptr,
        spatial_size,
        channel_size,
        channel_size / (norm_size / spatial_size));
    ChannelsLastStore<ComputeType, T, affine, FuseSwish> store(
        y_ptr,
        gamma_ptr,
        beta_ptr,
        spatial_size,
        channel_size,
        channel_size / (norm_size / spatial_size));

    layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
        stream,
        load,
        store,
        num_instances,
        norm_size,
        epsilon,
        mean,
        inv_variance);
  }
}

template <typename T, typename T2, bool FuseSwish>
void DispatchGroupNormForwardGpu(
    cudaStream_t stream,
    const int64_t num_instances,
    const int64_t norm_size,
    const int64_t channel_size,
    const int64_t spatial_size,
    const double epsilon,
    const T* x_ptr,
    const T* gamma_ptr,
    const T* beta_ptr,
    T* y_ptr,
    T2* mean,
    T2* inv_variance,
    bool channels_first) {
  using ComputeType = T2;
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    GroupNormForwardGpu<T, ComputeType, true, FuseSwish>(
        stream,
        num_instances,
        norm_size,
        channel_size,
        spatial_size,
        epsilon,
        x_ptr,
        gamma_ptr,
        beta_ptr,
        y_ptr,
        mean,
        inv_variance,
        channels_first);
  } else {
    GroupNormForwardGpu<T, ComputeType, false, FuseSwish>(
        stream,
        num_instances,
        norm_size,
        channel_size,
        spatial_size,
        epsilon,
        x_ptr,
        gamma_ptr,
        beta_ptr,
        y_ptr,
        mean,
        inv_variance,
        channels_first);
  }
}

template <typename TInput, bool FuseSwish, int H, int W, int C, int G>
cudaError_t invokeGroupNorm(
    TInput* output,
    TInput* input,
    TInput* gamma,
    TInput* beta,
    int N,
    const float eps,
    const int max_smem_size,
    void* workspace,
    cudaStream_t stream) {
  constexpr auto C_G = C / G;
  constexpr auto C_G_2 = C_G / 2;
  constexpr int ILP = 8;

  const int64_t num_instances = N * G;
  const int64_t norm_size = H * W * C / G;
  const int64_t spatial_size = H * W;
  const int64_t channel_size = C;
  const double epsilon = eps;
  bool channels_first = false;

  // Use a little bit more shared_memory to reduce occupancy and boost perf.
  constexpr int MEM_BANK_CONFLICT = 1;

  // Bank conflict doesn't seem to matter to perf
  constexpr int BANK_CONFLICT = 0;

  constexpr auto smem =
      H * W * (C_G_2 + MEM_BANK_CONFLICT) * 2 * sizeof(TInput);

  // C_G must be even, or we can have misaligned address for cp.async
  // reserve some shared_mem for block reduction
  if (H % 8 == 0 && C_G % 2 == 0 && smem <= max_smem_size - 1000) {
    constexpr int num_threads = std::min(1024, H / ILP * W * C_G_2);

    if constexpr (num_threads > 0) {
      auto kernel_func = group_norm_smem<
          TInput,
          FuseSwish,
          H,
          W,
          C,
          C_G,
          ILP,
          BANK_CONFLICT,
          num_threads>;
      GROUP_NORM_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      dim3 block(num_threads);
      kernel_func<<<dim3(G, N), block, smem, stream>>>(
          input, output, gamma, beta, N, eps);
    } else {
      DispatchGroupNormForwardGpu<TInput, float, FuseSwish>(
          stream,
          num_instances,
          norm_size,
          channel_size,
          spatial_size,
          epsilon,
          input,
          gamma,
          beta,
          output,
          static_cast<float*>(workspace),
          static_cast<float*>(workspace) + num_instances,
          channels_first);
    }
  } else {
    DispatchGroupNormForwardGpu<TInput, float, FuseSwish>(
        stream,
        num_instances,
        norm_size,
        channel_size,
        spatial_size,
        epsilon,
        input,
        gamma,
        beta,
        output,
        static_cast<float*>(workspace),
        static_cast<float*>(workspace) + num_instances,
        channels_first);
  }

  // GROUP_NORM_CUDA_CHECK_LAUNCH();
  // TODO: last error is 0, but invoked error logging no error
  return cudaGetLastError();
}

#endif /* GROUPNORM_KERNEL_CUH */
