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

// This kernel is based on CUTLASS LayerNorm kernel
#ifndef LAYERNORM_SIGMOID_MUL
#define LAYERNORM_SIGMOID_MUL

#define FINAL_MASK 0xffffffff

// TODO: can this header be used in ROCM with minimal changes?
#ifndef LAYER_NORM_CUDA_CHECK
#define LAYER_NORM_CUDA_CHECK(expr)                                       \
  do {                                                                    \
    cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " \
                << __FILE__ << ": " << __LINE__ << std::endl;             \
      return status;                                                      \
    }                                                                     \
  } while (0)
#endif

#ifndef LAYER_NORM_CUDA_CHECK_LAUNCH
#define LAYER_NORM_CUDA_CHECK_LAUNCH() LAYER_NORM_CUDA_CHECK(cudaGetLastError())
#endif

struct half4 {
  half x, y, z, w;
};

struct bfloat16_4 {
  bfloat16 x, y, z, w;
};

template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T* val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  // blockDim.x is round up to multiples of 32
  bool is_mask = threadIdx.x < (blockDim.x / 32);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}

template <typename T>
__inline__ __device__ T normalize(T val, T mean, T variance, T gamma, T beta) {
  return (val - mean) * variance * gamma + beta;
}

// __inline__ __device__ float sigmoid(float val) {
//   return 1.0f / (1.0f + expf(-1.0f * val));
// }

// fast sigmoid
__inline__ __device__ float sigmoid(float val) {
  return (cutlass::fast_tanh(val * 0.5f) + 1.0f) * 0.5f;
}

// output [m, n] row-major
// input [m, n] row-major
// gamma [n]
// beta [n]
// grid [m]
// block [block_size] -- each threadblock deals with block_size elements;
// block_size: round up to multiples of 32
// n = block_size
template <typename T, typename T_ACC, bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul_stored_locally(
    T* output,
    const T* input,
    const T* gamma,
    const T* beta,
    const int n,
    const T_ACC eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  __shared__ float s_mean, s_variance;
  const uint64_t offset = m_idx * n;

  float local_sums[1] = {0.0f};
  float local_val = 0.0f;
  if (tid < n) {
    local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + tid));
  }
  local_sums[0] = local_val;

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < n) {
    local_sums[0] = (local_val - s_mean) * (local_val - s_mean);
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[tid]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[tid]);
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    *(output_accessor.get<T, T>(output, offset + tid)) = T(local_val);
  }
}

// output [m, n] row-major
// input [m, n] row-major
// gamma [n]
// beta [n]
// grid [m]
// block [block_size] -- each threadblock deals with block_size elements;
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul_stored_locally(
    float4* output,
    const float4* input,
    const float4* gamma,
    const float4* beta,
    const int n,
    const float eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const uint64_t quarter_n = n >> 2;
  const uint64_t offset = m_idx * quarter_n;

  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};
  float local_sums[1] = {0.0f};
  if (tid < quarter_n) {
    local_val =
        *input_accessor.get<const float, const float4>(input, offset + tid);

    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const float4 gamma_val = gamma[tid];
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const float4 beta_val = beta[tid];
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    *(output_accessor.get<float, float4>(output, offset + tid)) = local_val;
  }
}

// output [m, n] row-major
// input [m, n] row-major
// gamma [n]
// beta [n]
// grid [m]
// block [block_size] -- each threadblock deals with block_size elements;
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul_stored_locally(
    half4* output,
    const half4* input,
    const half4* gamma,
    const half4* beta,
    const int n,
    const float eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const uint64_t quarter_n = n >> 2;
  const uint64_t offset = m_idx * quarter_n;

  float local_sums[1] = {0.0f};
  half4 local_val_half{0.0f, 0.0f, 0.0f, 0.0f};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  if (tid < quarter_n) {
    local_val_half =
        *input_accessor.get<const half, const half4>(input, offset + tid);

    local_val = {
        static_cast<float>(local_val_half.x),
        static_cast<float>(local_val_half.y),
        static_cast<float>(local_val_half.z),
        static_cast<float>(local_val_half.w)};
    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const half4 gamma_val_half = gamma[tid];
    const float4 gamma_val = {
        static_cast<float>(gamma_val_half.x),
        static_cast<float>(gamma_val_half.y),
        static_cast<float>(gamma_val_half.z),
        static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const half4 beta_val_half = beta[tid];
    const float4 beta_val = {
        static_cast<float>(beta_val_half.x),
        static_cast<float>(beta_val_half.y),
        static_cast<float>(beta_val_half.z),
        static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    local_val_half.x = __float2half_rn(local_val.x);
    local_val_half.y = __float2half_rn(local_val.y);
    local_val_half.z = __float2half_rn(local_val.z);
    local_val_half.w = __float2half_rn(local_val.w);

    *(output_accessor.get<half, half4>(output, offset + tid)) = local_val_half;
  }
}

// output [m, n] row-major
// input [m, n] row-major
// gamma [n]
// beta [n]
// grid [m]
// block [block_size] -- each threadblock deals with block_size elements;
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul_stored_locally(
    bfloat16_4* output,
    const bfloat16_4* input,
    const bfloat16_4* gamma,
    const bfloat16_4* beta,
    const int n,
    const float eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const uint64_t quarter_n = n >> 2;
  const uint64_t offset = m_idx * quarter_n;

  float local_sums[1] = {0.0f};
  bfloat16_4 local_val_half{0.0f, 0.0f, 0.0f, 0.0f};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  if (tid < quarter_n) {
    local_val_half = *input_accessor.get<const bfloat16, const bfloat16_4>(
        input, offset + tid);

    local_val = {
        static_cast<float>(local_val_half.x),
        static_cast<float>(local_val_half.y),
        static_cast<float>(local_val_half.z),
        static_cast<float>(local_val_half.w)};
    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const bfloat16_4 gamma_val_half = gamma[tid];
    const float4 gamma_val = {
        static_cast<float>(gamma_val_half.x),
        static_cast<float>(gamma_val_half.y),
        static_cast<float>(gamma_val_half.z),
        static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const bfloat16_4 beta_val_half = beta[tid];
    const float4 beta_val = {
        static_cast<float>(beta_val_half.x),
        static_cast<float>(beta_val_half.y),
        static_cast<float>(beta_val_half.z),
        static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    local_val_half.x = __float2bfloat16_rn(local_val.x);
    local_val_half.y = __float2bfloat16_rn(local_val.y);
    local_val_half.z = __float2bfloat16_rn(local_val.z);
    local_val_half.w = __float2bfloat16_rn(local_val.w);

    *(output_accessor.get<bfloat16, bfloat16_4>(output, offset + tid)) =
        local_val_half;
  }
}

// output [m, n] row-major
// input [m, n] row-major
// gamma [n]
// beta [n]
// grid(m)
// block(block_size) -- each thread deals with n / block_size elements
// block_size = 512
template <typename T, typename T_ACC, bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul(
    T* output,
    const T* input,
    const T* gamma,
    const T* beta,
    const int n,
    const T_ACC eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  __shared__ float s_mean, s_variance;
  const uint64_t offset = m_idx * n;

  float local_sums[1] = {0.0f};
  for (uint64_t i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));

    local_sums[0] += local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));

    local_sums[0] += (local_val - s_mean) * (local_val - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[i]);
#endif // AIT_LAYERNORM_CONST_GAMMA
#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[i]);
#endif // AIT_LAYERNORM_CONST_BETA
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));

    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    *(output_accessor.get<T, T>(output, offset + i)) = T(local_val);
  }
}

// Half specialization
template <bool FuseSigmoidMul>
__global__ void layernorm_sigmoid_mul(
    half* output,
    const half* input,
    const half* gamma,
    const half* beta,
    const int n,
    const float eps,
    TensorAccessor input_accessor,
    TensorAccessor output_accessor) {
  const uint64_t m_idx = blockIdx.x;
  const uint64_t tid = threadIdx.x;
  const uint64_t offset = m_idx * n;

  __shared__ float s_mean, s_variance;

  float local_sums[1] = {0.0f};
  for (uint64_t i = tid; i < n; i += blockDim.x) {
    float local_val = __half2float(
        *input_accessor.get<const half, const half>(input, offset + i));

    local_sums[0] += local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = __half2float(
        *input_accessor.get<const half, const half>(input, i + offset));

    local_sums[0] += (local_val - s_mean) * (local_val - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = __half2float(gamma[i]);
#endif // AIT_LAYERNORM_CONST_GAMMA
#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = __half2float(beta[i]);
#endif // AIT_LAYERNORM_CONST_BETA
    float local_val = __half2float(
        *input_accessor.get<const half, const half>(input, offset + i));

    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    *(output_accessor.get<half, half>(output, offset + i)) =
        __float2half_rn(local_val);
  }
}

template <typename T, typename T_ACC, bool FuseSigmoidMul>
cudaError_t invokeLayernormSigmoidMul(
    T* output,
    const T* input,
    const T* gamma,
    const T* beta,
    int m,
    int n,
    const T_ACC eps,
    cudaStream_t stream,
    const TensorAccessor& input_accessor,
    const TensorAccessor& output_accessor) {
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  dim3 grid(m);
  dim3 block(n);
  if ((n % 4 == 0) && (n >= 128) && (n <= 4096) &&
      /* float4 and half4 kernels read 4 elements at once;
         so they cannot be picked when an existing strided dim has the number of
         elements not divisible by 4 */
      input_accessor.is_valid_alignment(4) &&
      output_accessor.is_valid_alignment(4)) {
    block.x = (block.x / 4 + 31) / 32 * 32;
    if constexpr (std::is_same_v<T, float>) {
      layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (float4*)output,
              (const float4*)input,
              (const float4*)gamma,
              (const float4*)beta,
              n,
              eps,
              input_accessor,
              output_accessor);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    } else if constexpr (std::is_same_v<T, half>) {
      layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (half4*)output,
              (const half4*)input,
              (const half4*)gamma,
              (const half4*)beta,
              n,
              eps,
              input_accessor,
              output_accessor);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    } else if constexpr (std::is_same_v<T, bfloat16>) {
      layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (bfloat16_4*)output,
              (const bfloat16_4*)input,
              (const bfloat16_4*)gamma,
              (const bfloat16_4*)beta,
              n,
              eps,
              input_accessor,
              output_accessor);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    } else {
      static_assert(
          std::is_same_v<T, half> || std::is_same_v<T, float> ||
          std::is_same_v<T, bfloat16>);
    }
  } else if (n < 1024) {
    block.x = (block.x + 31) / 32 * 32;
    layernorm_sigmoid_mul_stored_locally<T, T_ACC, FuseSigmoidMul>
        <<<grid, block, 0, stream>>>(
            output,
            input,
            gamma,
            beta,
            n,
            eps,
            input_accessor,
            output_accessor);
    LAYER_NORM_CUDA_CHECK_LAUNCH();
  } else {
    CHECK(block.x >= 512);
    block.x = 512;
    if constexpr (std::is_same<T, half>::value) {
      layernorm_sigmoid_mul<FuseSigmoidMul><<<grid, block, 0, stream>>>(
          output, input, gamma, beta, n, eps, input_accessor, output_accessor);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    } else {
      layernorm_sigmoid_mul<T, T_ACC, FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              output,
              input,
              gamma,
              beta,
              n,
              eps,
              input_accessor,
              output_accessor);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    }
  }
  return cudaSuccess;
}

//================================BatchLayerNorm====================================

// output [b, m, n] row-major
// input [b, m, n] row-major
// gamma [b, n]
// beta [b, n]
// grid(b, m)
// block(block_size) -- each threadblock deals with block_size elements
// block_size: round up to multiples of 32
template <typename T, typename T_ACC, bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul_stored_locally(
    T* output,
    const T* input,
    const T* gamma,
    const T* beta,
    const int m,
    const int n,
    const T_ACC eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  // n is blocksize
  const int offset = (m_idx + b_idx * m) * n;
  const int gamma_beta_offset = b_idx * n;

  input += offset;
  output += offset;

  gamma += gamma_beta_offset;
  beta += gamma_beta_offset;

  float local_sums[1] = {0.0f};
  float local_val = 0.0f;
  if (tid < n) {
    local_val = static_cast<float>(input[tid]);
    ;
  }
  local_sums[0] = local_val;

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < n) {
    local_sums[0] = (local_val - s_mean) * (local_val - s_mean);
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[tid]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[tid]);
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    output[tid] = T(local_val);
  }
}

// output [b, m, n] row-major
// input [b, m, n] row-major
// gamma [b, n]
// beta [b, n]
// grid(b, m)
// block(block_size) -- each threadblock deals with block_size elements
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul_stored_locally(
    half4* output,
    const half4* input,
    const half4* gamma,
    const half4* beta,
    const int m,
    const int n,
    const float eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const int quarter_n = n >> 2;
  const int offset = (m_idx + b_idx * m) * quarter_n;
  const int gamma_beta_offset = b_idx * quarter_n;

  input += offset;
  output += offset;

  gamma += gamma_beta_offset;
  beta += gamma_beta_offset;

  float local_sums[1] = {0.0f};
  half4 local_val_half{0.0f, 0.0f, 0.0f, 0.0f};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  if (tid < quarter_n) {
    local_val_half = input[tid];
    local_val = {
        static_cast<float>(local_val_half.x),
        static_cast<float>(local_val_half.y),
        static_cast<float>(local_val_half.z),
        static_cast<float>(local_val_half.w)};
    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const half4 gamma_val_half = gamma[tid];
    const float4 gamma_val = {
        static_cast<float>(gamma_val_half.x),
        static_cast<float>(gamma_val_half.y),
        static_cast<float>(gamma_val_half.z),
        static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const half4 beta_val_half = beta[tid];
    const float4 beta_val = {
        static_cast<float>(beta_val_half.x),
        static_cast<float>(beta_val_half.y),
        static_cast<float>(beta_val_half.z),
        static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    local_val_half.x = __float2half_rn(local_val.x);
    local_val_half.y = __float2half_rn(local_val.y);
    local_val_half.z = __float2half_rn(local_val.z);
    local_val_half.w = __float2half_rn(local_val.w);

    output[tid] = local_val_half;
  }
}

// output [b, m, n] row-major
// input [b, m, n] row-major
// gamma [b, n]
// beta [b, n]
// grid(b, m)
// block(block_size) -- each threadblock deals with block_size elements
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul_stored_locally(
    bfloat16_4* output,
    const bfloat16_4* input,
    const bfloat16_4* gamma,
    const bfloat16_4* beta,
    const int m,
    const int n,
    const float eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const int quarter_n = n >> 2;
  const int offset = (m_idx + b_idx * m) * quarter_n;
  const int gamma_beta_offset = b_idx * quarter_n;

  input += offset;
  output += offset;

  gamma += gamma_beta_offset;
  beta += gamma_beta_offset;

  float local_sums[1] = {0.0f};
  bfloat16_4 local_val_half{0.0f, 0.0f, 0.0f, 0.0f};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  if (tid < quarter_n) {
    local_val_half = input[tid];
    local_val = {
        static_cast<float>(local_val_half.x),
        static_cast<float>(local_val_half.y),
        static_cast<float>(local_val_half.z),
        static_cast<float>(local_val_half.w)};
    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const bfloat16_4 gamma_val_half = gamma[tid];
    const float4 gamma_val = {
        static_cast<float>(gamma_val_half.x),
        static_cast<float>(gamma_val_half.y),
        static_cast<float>(gamma_val_half.z),
        static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const bfloat16_4 beta_val_half = beta[tid];
    const float4 beta_val = {
        static_cast<float>(beta_val_half.x),
        static_cast<float>(beta_val_half.y),
        static_cast<float>(beta_val_half.z),
        static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    local_val_half.x = __float2bfloat16_rn(local_val.x);
    local_val_half.y = __float2bfloat16_rn(local_val.y);
    local_val_half.z = __float2bfloat16_rn(local_val.z);
    local_val_half.w = __float2bfloat16_rn(local_val.w);

    output[tid] = local_val_half;
  }
}

// output [b, m, n] row-major
// input [b, m, n] row-major
// gamma [b, n]
// beta [b, n]
// grid(b, m)
// block(block_size) -- each threadblock deals with block_size elements
// block_size = n / 4
// block_size: round up to multiples of 32
template <bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul_stored_locally(
    float4* output,
    const float4* input,
    const float4* gamma,
    const float4* beta,
    const int m,
    const int n,
    const float eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const int quarter_n = n >> 2;
  const int offset = (m_idx + b_idx * m) * quarter_n;

  input += offset;
  output += offset;

#if !defined(AIT_LAYERNORM_CONST_GAMMA) || !defined(AIT_LAYERNORM_CONST_BETA)
  const int gamma_beta_offset = b_idx * quarter_n;
#endif // !AIT_LAYERNORM_CONST_GAMMA || !AIT_LAYERNORM_CONST_BETA

#ifndef AIT_LAYERNORM_CONST_GAMMA
  gamma += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifndef AIT_LAYERNORM_CONST_BETA
  beta += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_BETA

  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};
  float local_sums[1] = {0.0f};
  if (tid < quarter_n) {
    local_val = input[tid];
    local_sums[0] = local_val.x + local_val.y + local_val.z + local_val.w;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < quarter_n) {
    local_sums[0] = (local_val.x - s_mean) * (local_val.x - s_mean) +
        (local_val.y - s_mean) * (local_val.y - s_mean) +
        (local_val.z - s_mean) * (local_val.z - s_mean) +
        (local_val.w - s_mean) * (local_val.w - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  if (tid < quarter_n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float4 gamma_val = {
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA,
        AIT_LAYERNORM_CONST_GAMMA};
#else
    const float4 gamma_val = gamma[tid];
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float4 beta_val = {
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA,
        AIT_LAYERNORM_CONST_BETA};
#else
    const float4 beta_val = beta[tid];
#endif // AIT_LAYERNORM_CONST_BETA

    if (FuseSigmoidMul) {
      local_val.x *= sigmoid(
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
      local_val.y *= sigmoid(
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
      local_val.z *= sigmoid(
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
      local_val.w *= sigmoid(
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
    } else {
      local_val.x =
          normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
      local_val.y =
          normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
      local_val.z =
          normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
      local_val.w =
          normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
    }

    output[tid] = local_val;
  }
}

// output [b, m, n] row-major
// input [b, m, n] row-major
// gamma [b, n]
// beta [b, n]
// grid(b, m)
// block(block_size) -- each thread deals with n / block_size elements
// block_size = 512
template <typename T, typename T_ACC, bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul(
    T* output,
    T* input,
    const T* gamma,
    const T* beta,
    const int m,
    const int n,
    const T_ACC eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const int offset = (m_idx + b_idx * m) * n;
#if !defined(AIT_LAYERNORM_CONST_GAMMA) || !defined(AIT_LAYERNORM_CONST_BETA)
  const int gamma_beta_offset = b_idx * n;
#endif // !AIT_LAYERNORM_CONST_GAMMA || !AIT_LAYERNORM_CONST_BETA

  input += offset;
  output += offset;

#ifndef AIT_LAYERNORM_CONST_GAMMA
  gamma += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifndef AIT_LAYERNORM_CONST_BETA
  beta += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_BETA

  float local_sums[1] = {0.0f};
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(input[i]);
    local_sums[0] += local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(input[i]);
    local_sums[0] += (local_val - s_mean) * (local_val - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[i]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[i]);
#endif // AIT_LAYERNORM_CONST_BETA

    float local_val = static_cast<float>(input[i]);
    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    output[i] = T(local_val);
  }
}

// half specialization
template <bool FuseSigmoidMul>
__global__ void batch_layernorm_sigmoid_mul(
    half* output,
    half* input,
    const half* gamma,
    const half* beta,
    const int m,
    const int n,
    const float eps) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  const int offset = (m_idx + b_idx * m) * n;
#if !defined(AIT_LAYERNORM_CONST_GAMMA) || !defined(AIT_LAYERNORM_CONST_BETA)
  const int gamma_beta_offset = b_idx * n;
#endif // !AIT_LAYERNORM_CONST_GAMMA || !AIT_LAYERNORM_CONST_BETA

  input += offset;
  output += offset;

#ifndef AIT_LAYERNORM_CONST_GAMMA
  gamma += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifndef AIT_LAYERNORM_CONST_BETA
  beta += gamma_beta_offset;
#endif // AIT_LAYERNORM_CONST_BETA

  float local_sums[1] = {0.0f};
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = __half2float(input[i]);
    local_sums[0] += local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = __half2float(input[i]);
    local_sums[0] += (local_val - s_mean) * (local_val - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = __half2float(gamma[i]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = __half2float(beta[i]);
#endif // AIT_LAYERNORM_CONST_BETA
    float local_val = __half2float(input[i]);
    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    output[i] = __float2half_rn(local_val);
  }
}

template <typename T, typename T_ACC, bool FuseSigmoidMul>
void invokeBatchLayernormSigmoidMul(
    T* output,
    T* input,
    const T* gamma,
    const T* beta,
    int b,
    int m,
    int n,
    const T_ACC eps,
    cudaStream_t stream) {
  if (b == 0 || m == 0 || n == 0) {
    return;
  }
  dim3 grid(b, m);
  dim3 block(n);
  if ((n % 4 == 0) && (n >= 128) && (n <= 4096)) {
    block.x = (block.x / 4 + 31) / 32 * 32;
    if constexpr (std::is_same<T, float>::value) {
      batch_layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (float4*)output,
              (const float4*)input,
              (const float4*)gamma,
              (const float4*)beta,
              m,
              n,
              eps);
    } else if constexpr (std::is_same<T, half>::value) {
      batch_layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (half4*)output,
              (const half4*)input,
              (const half4*)gamma,
              (const half4*)beta,
              m,
              n,
              eps);
    } else if constexpr (std::is_same<T, bfloat16>::value) {
      batch_layernorm_sigmoid_mul_stored_locally<FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(
              (bfloat16_4*)output,
              (const bfloat16_4*)input,
              (const bfloat16_4*)gamma,
              (const bfloat16_4*)beta,
              m,
              n,
              eps);
    } else {
      static_assert(
          std::is_same_v<T, half> || std::is_same_v<T, float> ||
          std::is_same_v<T, bfloat16>);
    }
  } else if (n < 1024) {
    block.x = (block.x + 31) / 32 * 32;
    batch_layernorm_sigmoid_mul_stored_locally<T, T_ACC, FuseSigmoidMul>
        <<<grid, block, 0, stream>>>(output, input, gamma, beta, m, n, eps);
  } else {
    CHECK(block.x >= 512);
    block.x = 512;
    if (std::is_same<T, half>::value) {
      batch_layernorm_sigmoid_mul<FuseSigmoidMul><<<grid, block, 0, stream>>>(
          (half*)(output),
          (half*)(input),
          (const half*)(gamma),
          (const half*)(beta),
          m,
          n,
          eps);
    } else {
      batch_layernorm_sigmoid_mul<T, T_ACC, FuseSigmoidMul>
          <<<grid, block, 0, stream>>>(output, input, gamma, beta, m, n, eps);
    }
  }
}

//================================GroupLayerNorm====================================

template <typename T, typename T_ACC, int NumInputs>
struct Arguments {
  T* outputs[NumInputs]; /* pointer to each output */
  T* inputs[NumInputs]; /* pointer to each input */
  T* gammas[NumInputs];
  T* betas[NumInputs];
  int64_t N[NumInputs]; /* N of each input */
  T_ACC eps;
  TensorAccessor input_accessors[NumInputs];
  TensorAccessor output_accessors[NumInputs];
};

// output b * [m, n] row-major
// input  b * [m, n] row-major
// gamma b * [n]
// beta  b * [n]
// grid [b, m]
// block [block_size] -- each thread deals with 4 elements
// block_size = n / 4
template <bool FuseSigmoidMul, int NumInputs>
__device__ void group_layernorm_sigmoid_mul_stored_locally_impl(
    const Arguments<half4, float, NumInputs>& args) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};

  half4* output = args.outputs[b_idx];
  const half4* input = args.inputs[b_idx];
  const half4* gamma = args.gammas[b_idx];
  const half4* beta = args.betas[b_idx];
  const TensorAccessor& input_accessor = args.input_accessors[b_idx];
  const TensorAccessor& output_accessor = args.output_accessors[b_idx];

  const int n = args.N[b_idx];
  const int quarter_n = n >> 2;
  const int offset = m_idx * quarter_n;

  const int block_size = blockDim.x;
  const int num_iters =
      ceil(static_cast<float>(quarter_n) / static_cast<float>(block_size));

  half4 local_val_half{0, 0, 0, 0};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;

    if (elem_no < quarter_n) {
      local_val_half =
          *input_accessor.get<const half, const half4>(input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
      local_sums[0] += local_val.x + local_val.y + local_val.z + local_val.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();
  local_sums[0] = 0.0f;

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;
    if (elem_no < quarter_n) {
      local_val_half =
          *input_accessor.get<const half, const half4>(input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
      local_sums[0] += (local_val.x - s_mean) * (local_val.x - s_mean) +
          (local_val.y - s_mean) * (local_val.y - s_mean) +
          (local_val.z - s_mean) * (local_val.z - s_mean) +
          (local_val.w - s_mean) * (local_val.w - s_mean);
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + args.eps);
  }
  __syncthreads();

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;
    if (elem_no < quarter_n) {
      local_val_half =
          *input_accessor.get<const half, const half4>(input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
#ifdef AIT_LAYERNORM_CONST_GAMMA
      const float4 gamma_val = {
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA};
#else
      const half4 gamma_val_half = gamma[elem_no];
      const float4 gamma_val = {
          static_cast<float>(gamma_val_half.x),
          static_cast<float>(gamma_val_half.y),
          static_cast<float>(gamma_val_half.z),
          static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
      const float4 beta_val = {
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA};
#else
      const half4 beta_val_half = beta[elem_no];
      const float4 beta_val = {
          static_cast<float>(beta_val_half.x),
          static_cast<float>(beta_val_half.y),
          static_cast<float>(beta_val_half.z),
          static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

      if constexpr (FuseSigmoidMul) {
        local_val.x *= sigmoid(normalize(
            local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
        local_val.y *= sigmoid(normalize(
            local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
        local_val.z *= sigmoid(normalize(
            local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
        local_val.w *= sigmoid(normalize(
            local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
      } else {
        local_val.x =
            normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
        local_val.y =
            normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
        local_val.z =
            normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
        local_val.w =
            normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
      }

      local_val_half.x = __float2half_rn(local_val.x);
      local_val_half.y = __float2half_rn(local_val.y);
      local_val_half.z = __float2half_rn(local_val.z);
      local_val_half.w = __float2half_rn(local_val.w);

      *(output_accessor.get<half, half4>(output, offset + elem_no)) =
          local_val_half;
    }
  }
}

// output b * [m, n] row-major
// input  b * [m, n] row-major
// gamma b * [n]
// beta  b * [n]
// grid [b, m]
// block [block_size] -- each thread deals with 4 elements
// block_size = n / 4
template <bool FuseSigmoidMul, int NumInputs>
__device__ void group_layernorm_sigmoid_mul_stored_locally_impl(
    const Arguments<bfloat16_4, float, NumInputs>& args) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};

  bfloat16_4* output = args.outputs[b_idx];
  const bfloat16_4* input = args.inputs[b_idx];
  const bfloat16_4* gamma = args.gammas[b_idx];
  const bfloat16_4* beta = args.betas[b_idx];
  const TensorAccessor& input_accessor = args.input_accessors[b_idx];
  const TensorAccessor& output_accessor = args.output_accessors[b_idx];

  const int n = args.N[b_idx];
  const int quarter_n = n >> 2;
  const int offset = m_idx * quarter_n;

  const int block_size = blockDim.x;
  const int num_iters =
      ceil(static_cast<float>(quarter_n) / static_cast<float>(block_size));

  bfloat16_4 local_val_half{0, 0, 0, 0};
  float4 local_val{0.0f, 0.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;

    if (elem_no < quarter_n) {
      local_val_half = *input_accessor.get<const bfloat16, const bfloat16_4>(
          input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
      local_sums[0] += local_val.x + local_val.y + local_val.z + local_val.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();
  local_sums[0] = 0.0f;

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;
    if (elem_no < quarter_n) {
      local_val_half = *input_accessor.get<const bfloat16, const bfloat16_4>(
          input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
      local_sums[0] += (local_val.x - s_mean) * (local_val.x - s_mean) +
          (local_val.y - s_mean) * (local_val.y - s_mean) +
          (local_val.z - s_mean) * (local_val.z - s_mean) +
          (local_val.w - s_mean) * (local_val.w - s_mean);
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + args.eps);
  }
  __syncthreads();

  for (size_t i = 0; i < num_iters; ++i) {
    int elem_no = tid + block_size * i;
    if (elem_no < quarter_n) {
      local_val_half = *input_accessor.get<const bfloat16, const bfloat16_4>(
          input, offset + elem_no);
      local_val = {
          static_cast<float>(local_val_half.x),
          static_cast<float>(local_val_half.y),
          static_cast<float>(local_val_half.z),
          static_cast<float>(local_val_half.w)};
#ifdef AIT_LAYERNORM_CONST_GAMMA
      const float4 gamma_val = {
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA,
          AIT_LAYERNORM_CONST_GAMMA};
#else
      const bfloat16_4 gamma_val_half = gamma[elem_no];
      const float4 gamma_val = {
          static_cast<float>(gamma_val_half.x),
          static_cast<float>(gamma_val_half.y),
          static_cast<float>(gamma_val_half.z),
          static_cast<float>(gamma_val_half.w)};
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
      const float4 beta_val = {
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA,
          AIT_LAYERNORM_CONST_BETA};
#else
      const bfloat16_4 beta_val_half = beta[elem_no];
      const float4 beta_val = {
          static_cast<float>(beta_val_half.x),
          static_cast<float>(beta_val_half.y),
          static_cast<float>(beta_val_half.z),
          static_cast<float>(beta_val_half.w)};
#endif // AIT_LAYERNORM_CONST_BETA

      if constexpr (FuseSigmoidMul) {
        local_val.x *= sigmoid(normalize(
            local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x));
        local_val.y *= sigmoid(normalize(
            local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y));
        local_val.z *= sigmoid(normalize(
            local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z));
        local_val.w *= sigmoid(normalize(
            local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w));
      } else {
        local_val.x =
            normalize(local_val.x, s_mean, s_variance, gamma_val.x, beta_val.x);
        local_val.y =
            normalize(local_val.y, s_mean, s_variance, gamma_val.y, beta_val.y);
        local_val.z =
            normalize(local_val.z, s_mean, s_variance, gamma_val.z, beta_val.z);
        local_val.w =
            normalize(local_val.w, s_mean, s_variance, gamma_val.w, beta_val.w);
      }

      local_val_half.x = __float2bfloat16_rn(local_val.x);
      local_val_half.y = __float2bfloat16_rn(local_val.y);
      local_val_half.z = __float2bfloat16_rn(local_val.z);
      local_val_half.w = __float2bfloat16_rn(local_val.w);

      *(output_accessor.get<bfloat16, bfloat16_4>(output, offset + elem_no)) =
          local_val_half;
    }
  }
}

#define GROUP_LAYER_NORM_MAX_INLINE_INPUTS 39

template <
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS, bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally_half(
    Arguments<half4, float, NumInputs> args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<FuseSigmoidMul, NumInputs>(
      args);
}

template <
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<(NumInputs > GROUP_LAYER_NORM_MAX_INLINE_INPUTS), bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally_half(
    const Arguments<half4, float, NumInputs>* args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<FuseSigmoidMul, NumInputs>(
      *args);
}

template <
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS, bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally_bfloat16(
    Arguments<bfloat16_4, float, NumInputs> args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<FuseSigmoidMul, NumInputs>(
      args);
}

template <
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<(NumInputs > GROUP_LAYER_NORM_MAX_INLINE_INPUTS), bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally_bfloat16(
    const Arguments<bfloat16_4, float, NumInputs>* args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<FuseSigmoidMul, NumInputs>(
      *args);
}

// output b * [m, n] row-major
// input  b * [m, n] row-major
// gamma b * [n]
// beta  b * [n]
// grid [b, m]
// block [block_size] -- each thread deals with 1 element
// block_size = n
template <typename T, typename T_ACC, bool FuseSigmoidMul, int NumInputs>
__device__ void group_layernorm_sigmoid_mul_stored_locally_impl(
    const Arguments<T, T_ACC, NumInputs>& args) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  T* output = args.outputs[b_idx];
  const T* input = args.inputs[b_idx];
  const T* gamma = args.gammas[b_idx];
  const T* beta = args.betas[b_idx];
  const TensorAccessor& input_accessor = args.input_accessors[b_idx];
  const TensorAccessor& output_accessor = args.output_accessors[b_idx];

  const int n = args.N[b_idx];
  const int offset = m_idx * n;

  float local_sums[1] = {0.0f};
  float local_val = 0.0f;

  if (tid < n) {
    local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + tid));
    local_sums[0] = local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  if (tid < n) {
    local_sums[0] = (local_val - s_mean) * (local_val - s_mean);
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + args.eps);
  }
  __syncthreads();

  if (tid < n) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[tid]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[tid]);
#endif // AIT_LAYERNORM_CONST_BETA

    if constexpr (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    *(output_accessor.get<T, T>(output, offset + tid)) = T(local_val);
  }
}

template <
    typename T,
    typename T_ACC,
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS, bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally(
    Arguments<T, T_ACC, NumInputs> args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<
      T,
      T_ACC,
      FuseSigmoidMul,
      NumInputs>(args);
}

template <
    typename T,
    typename T_ACC,
    bool FuseSigmoidMul,
    int NumInputs,
    std::enable_if_t<(NumInputs > GROUP_LAYER_NORM_MAX_INLINE_INPUTS), bool> =
        true>
__global__ void group_layernorm_sigmoid_mul_stored_locally(
    const Arguments<T, T_ACC, NumInputs>* args) {
  group_layernorm_sigmoid_mul_stored_locally_impl<
      T,
      T_ACC,
      FuseSigmoidMul,
      NumInputs>(*args);
}

// output b * [m, n] row-major
// input  b * [m, n] row-major
// gamma b * [n]
// beta  b * [n]
// grid [b, m]
// block [block_size] -- each thread deals with n / block_size element
// block_size = 512
template <typename T, typename T_ACC, bool FuseSigmoidMul, int NumInputs>
__device__ void group_layernorm_sigmoid_mul_impl(
    Arguments<T, T_ACC, NumInputs> args) {
  const int b_idx = blockIdx.x;
  const int m_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float s_mean, s_variance;

  T* output = args.outputs[b_idx];
  const T* input = args.inputs[b_idx];
  const T* gamma = args.gammas[b_idx];
  const T* beta = args.betas[b_idx];
  const TensorAccessor& input_accessor = args.input_accessors[b_idx];
  const TensorAccessor& output_accessor = args.output_accessors[b_idx];

  const int n = args.N[b_idx];
  int offset = m_idx * n;

  float local_sums[1] = {0.0f};
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));
    local_sums[0] += local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));
    local_sums[0] += (local_val - s_mean) * (local_val - s_mean);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + args.eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
#ifdef AIT_LAYERNORM_CONST_GAMMA
    const float gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
    const float gamma_val = static_cast<float>(gamma[i]);
#endif // AIT_LAYERNORM_CONST_GAMMA
#ifdef AIT_LAYERNORM_CONST_BETA
    const float beta_val = AIT_LAYERNORM_CONST_BETA;
#else
    const float beta_val = static_cast<float>(beta[i]);
#endif // AIT_LAYERNORM_CONST_BETA
    float local_val = static_cast<float>(
        *input_accessor.get<const T, const T>(input, offset + i));
    if (FuseSigmoidMul) {
      local_val *= sigmoid(
          normalize(local_val, s_mean, s_variance, gamma_val, beta_val));
    } else {
      local_val = normalize(local_val, s_mean, s_variance, gamma_val, beta_val);
    }

    *(output_accessor.get<T, T>(output, offset + i)) = T(local_val);
  }
}

template <typename T, typename T_ACC, bool FuseSigmoidMul, int NumInputs>
__global__ void group_layernorm_sigmoid_mul(
    Arguments<T, T_ACC, NumInputs> args) {
  group_layernorm_sigmoid_mul_impl<T, T_ACC, FuseSigmoidMul, NumInputs>(args);
}

template <typename T, typename T_ACC, bool FuseSigmoidMul, int NumInputs>
__global__ void group_layernorm_sigmoid_mul(
    const Arguments<T, T_ACC, NumInputs>* args) {
  group_layernorm_sigmoid_mul_impl<T, T_ACC, FuseSigmoidMul, NumInputs>(*args);
}

// array size of output, input, gamma, beta, n: b (group size)
template <typename T, typename T_ACC, bool FuseSigmoidMul, int NumInputs>
cudaError_t invokeGroupLayernormSigmoidMul(
    T* output[],
    T* input[],
    T* gamma[],
    T* beta[],
    int b,
    int m,
    const int64_t* n,
    const T_ACC eps,
    cudaStream_t stream,
    const TensorAccessor* input_accessors,
    const TensorAccessor* output_accessors) {
  bool n_is_multiple_of_4 =
      std::all_of(n, n + b, [](int i) { return i % 4 == 0; });

  int max_n = *std::max_element(n, n + b);
  int min_n = *std::min_element(n, n + b);
  if (max_n == 0) {
    return cudaSuccess;
  }

  dim3 grid(b, m);
  // TODO: implement float4 group kernel
  if (std::is_same<T, half>::value && n_is_multiple_of_4 && (min_n >= 128) &&
      (max_n <= 4096)) {
    dim3 block(min_n);
    // round up to multiples of 32 to make warp shuffles safe
    block.x = (block.x / 4 + 31) / 32 * 32;
    Arguments<half4, float, NumInputs> args;
    for (size_t i = 0; i < b; i++) {
      args.outputs[i] = reinterpret_cast<half4*>(output[i]);
      args.inputs[i] = reinterpret_cast<half4*>(input[i]);
      args.gammas[i] = reinterpret_cast<half4*>(gamma[i]);
      args.betas[i] = reinterpret_cast<half4*>(beta[i]);
      args.N[i] = n[i];
      args.eps = eps;
      args.output_accessors[i] = output_accessors[i];
      args.input_accessors[i] = input_accessors[i];
    }
    if constexpr (NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS) {
      group_layernorm_sigmoid_mul_stored_locally_half<FuseSigmoidMul, NumInputs>
          <<<grid, block, 0, stream>>>(args);
      LAYER_NORM_CUDA_CHECK_LAUNCH();
    } else {
      Arguments<half4, float, NumInputs>* argsPtr;
      LAYER_NORM_CUDA_CHECK(cudaMalloc(&argsPtr, sizeof(args)));
      LAYER_NORM_CUDA_CHECK(
          cudaMemcpy(argsPtr, &args, sizeof(args), cudaMemcpyHostToDevice));
      group_layernorm_sigmoid_mul_stored_locally_half<FuseSigmoidMul, NumInputs>
          <<<grid, block, 0, stream>>>(
              const_cast<const Arguments<half4, float, NumInputs>*>(argsPtr));
      LAYER_NORM_CUDA_CHECK_LAUNCH();
      LAYER_NORM_CUDA_CHECK(cudaFree(argsPtr));
    }
  } else {
    // TODO: Should we apply min_n block size to this branch as well?
    dim3 block(max_n);
    Arguments<T, T_ACC, NumInputs> args;
    for (size_t i = 0; i < b; i++) {
      args.outputs[i] = output[i];
      args.inputs[i] = input[i];
      args.gammas[i] = gamma[i];
      args.betas[i] = beta[i];
      args.N[i] = n[i];
      args.eps = eps;
      args.input_accessors[i] = input_accessors[i];
      args.output_accessors[i] = output_accessors[i];
    }
    if (max_n < 1024) {
      block.x = (block.x + 31) / 32 * 32;
      if constexpr (NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS) {
        group_layernorm_sigmoid_mul_stored_locally<
            T,
            T_ACC,
            FuseSigmoidMul,
            NumInputs><<<grid, block, 0, stream>>>(args);
        LAYER_NORM_CUDA_CHECK_LAUNCH();
      } else {
        Arguments<T, T_ACC, NumInputs>* argsPtr;
        LAYER_NORM_CUDA_CHECK(cudaMalloc(&argsPtr, sizeof(args)));
        LAYER_NORM_CUDA_CHECK(
            cudaMemcpy(argsPtr, &args, sizeof(args), cudaMemcpyHostToDevice));
        group_layernorm_sigmoid_mul_stored_locally<
            T,
            T_ACC,
            FuseSigmoidMul,
            NumInputs><<<grid, block, 0, stream>>>(argsPtr);
        LAYER_NORM_CUDA_CHECK_LAUNCH();
        LAYER_NORM_CUDA_CHECK(cudaFree(argsPtr));
      }
    } else {
      CHECK(block.x >= 512);
      block.x = 512;
      if constexpr (NumInputs <= GROUP_LAYER_NORM_MAX_INLINE_INPUTS) {
        group_layernorm_sigmoid_mul<T, T_ACC, FuseSigmoidMul, NumInputs>
            <<<grid, block, 0, stream>>>(args);
        LAYER_NORM_CUDA_CHECK_LAUNCH();
      } else {
        Arguments<T, T_ACC, NumInputs>* argsPtr;
        LAYER_NORM_CUDA_CHECK(cudaMalloc(&argsPtr, sizeof(args)));
        LAYER_NORM_CUDA_CHECK(
            cudaMemcpy(argsPtr, &args, sizeof(args), cudaMemcpyHostToDevice));
        group_layernorm_sigmoid_mul<T, T_ACC, FuseSigmoidMul, NumInputs>
            <<<grid, block, 0, stream>>>(argsPtr);
        LAYER_NORM_CUDA_CHECK_LAUNCH();
        LAYER_NORM_CUDA_CHECK(cudaFree(argsPtr));
      }
    }
  }
  return cudaSuccess;
}

#undef LAYER_NORM_CUDA_CHECK
#undef LAYER_NORM_CUDA_CHECK_LAUNCH

#endif
