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

#define FINAL_MASK 0xffffffff

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

__inline__ __device__ float sigmoid(float val) {
  return (cutlass::fast_tanh(val * 0.5f) + 1.0f) * 0.5f;
}

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

#undef __AIT_GN_USE_FAST_MATH

template <typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, int* count) {
  // Use Welford Online algorithem to compute mean and variance
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
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    int b_count =
        __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
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

template <
    bool FuseSwish,
    int H,
    int W,
    int C,
    int C_G,
    int ILP = 8,
    int BANK_CONFLICT = 0,
    int NUM_THREADS = 1024>
__global__ __launch_bounds__(NUM_THREADS) void group_norm_smem(
    const half* X,
    half* Y,
    half* gamma,
    half* beta,
    int N,
    float epsilon) {
  constexpr int C_G_2 = C_G / 2;
  constexpr int C_G_stride = C_G_2 + BANK_CONFLICT;
  extern __shared__ int svals_[];
  auto* svals = reinterpret_cast<__half2*>(&svals_[0]);

  int32_t g = blockIdx.x;
  int32_t start_c = g * C_G;
  int32_t n = blockIdx.y;

  // X: [N, H, W, C]
  int32_t strides[4] = {H * W * C, W * C, C, 1};
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    auto c_g_2 = load_idx % C_G_2;
    auto w = (load_idx / C_G_2) % W;
    auto h_ilp = ((load_idx / C_G_2) / W);

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      const __half2* src = reinterpret_cast<const __half2*>(
          &(X[n * strides[0] + (h_ilp * ILP + ii) * strides[1] +
              w * strides[2] + (start_c + c_g_2 * 2)]));
      __half2* dst =
          &svals[(h_ilp * ILP + ii) * W * C_G_stride + w * C_G_stride + c_g_2];
      cutlass::arch::cp_async_zfill<sizeof(__half2)>(dst, src, true);
    }
  }
  cutlass::arch::cp_async_wait<0>();

  float thread_sum = 0;
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    auto c_g_2 = load_idx % C_G_2;
    auto w = (load_idx / C_G_2) % W;
    auto h_ilp = ((load_idx / C_G_2) / W);
#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      half2 valh =
          svals[(h_ilp * ILP + ii) * W * C_G_stride + w * C_G_stride + c_g_2];
      float2 val = __half22float2(valh);
      thread_sum += val.x + val.y;
    }
  }
  const float block_mean =
      BlockAllReduce<SumOp, float, NUM_THREADS>(thread_sum) /
      float(H * W * C_G);

  float thread_sq_sum = 0;
  for (int32_t load_idx = threadIdx.x; load_idx < H / ILP * W * C_G_2;
       load_idx += blockDim.x) {
    auto c_g_2 = load_idx % C_G_2;
    auto w = (load_idx / C_G_2) % W;
    auto h_ilp = ((load_idx / C_G_2) / W);

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      half2 valh =
          svals[(h_ilp * ILP + ii) * W * C_G_stride + w * C_G_stride + c_g_2];
      float2 val = __half22float2(valh);
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
    auto c_g_2 = load_idx % C_G_2;
    auto w = (load_idx / C_G_2) % W;
    auto h_ilp = ((load_idx / C_G_2) / W);

    auto g = __half22float2(
        *reinterpret_cast<const __half2*>(&gamma[start_c + c_g_2 * 2]));
    g.x *= block_inv_std;
    g.y *= block_inv_std;
    auto b = __half22float2(
        *reinterpret_cast<const __half2*>(&beta[start_c + c_g_2 * 2]));

#pragma unroll ILP
    for (auto ii = 0; ii < ILP; ++ii) {
      __half2* src =
          &svals[(h_ilp * ILP + ii) * W * C_G_stride + w * C_G_stride + c_g_2];
      __half2* dst = reinterpret_cast<__half2*>(
          &(Y[n * strides[0] + (h_ilp * ILP + ii) * strides[1] +
              w * strides[2] + (start_c + c_g_2 * 2)]));

      auto fsrc = __half22float2(*src);
      float2 result;
      result.x = (fsrc.x - block_mean) * g.x + b.x;
      result.y = (fsrc.y - block_mean) * g.y + b.y;
      if (FuseSwish) {
        result.x = result.x * sigmoid(result.x);
        result.y = result.y * sigmoid(result.y);
      }
      *dst = __float22half2_rn(result);
    }
  }
}

template <bool FuseSwish, int H, int W, int C, int num_groups>
cudaError_t invokeWelfordGroupNorm(
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

template <bool FuseSwish, int H, int W, int C, int G>
cudaError_t invokeGroupNorm(
    half* output,
    half* input,
    half* gamma,
    half* beta,
    int N,
    const float eps,
    const int max_smem_size,
    cudaStream_t stream) {
  constexpr auto C_G = C / G;
  constexpr auto C_G_2 = C_G / 2;
  constexpr int ILP = 8;

  // Use a little big more shared_memory to reduce occupancy and boost perf.
  constexpr int MEM_BANK_CONFLICT = 1;

  // Bank conflict doesn't seem to matter to perf
  constexpr int BANK_CONFLICT = 0;

  const auto smem = H * W * (C_G_2 + MEM_BANK_CONFLICT) * 2 * sizeof(uint16_t);

  // C_G must be even, or we can have misaligned address for cp.async
  // reserve some shared_mem for block reduction
  if (H % 8 == 0 && C_G % 2 == 0 && smem <= max_smem_size - 1000) {
    GROUP_NORM_CUDA_CHECK(cudaFuncSetAttribute(
        group_norm_smem<FuseSwish, H, W, C, C_G, ILP, BANK_CONFLICT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));

    constexpr int num_threads = std::min(1024, H / ILP * W * C_G_2);

    dim3 block(num_threads);
    group_norm_smem<FuseSwish, H, W, C, C_G, ILP, BANK_CONFLICT, num_threads>
        <<<dim3(G, N), block, smem, stream>>>(
            input, output, gamma, beta, N, eps);
  } else {
    return invokeWelfordGroupNorm<FuseSwish, H, W, C, G>(
        output, input, gamma, beta, N, eps, stream);
  }

  // GROUP_NORM_CUDA_CHECK_LAUNCH();
  // TODO: last error is 0, but invoked error logging no error
  return cudaGetLastError();
}

#endif /* GROUPNORM_KERNEL_CUH */
