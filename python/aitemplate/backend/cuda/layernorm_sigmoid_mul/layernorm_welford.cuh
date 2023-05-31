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
#ifndef LAYERNORM_KERNEL_CUH
#define LAYERNORM_KERNEL_CUH

constexpr uint32_t kFinalMask = 0xffffffff;

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#endif

#define NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)

__device__ half fast_tanh(half x) {
#if defined(AIT_USE_FAST_MATH)
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 750)

  asm volatile("tanh.approx.f16 %0, %1;"
               : "=h"(__HALF_TO_US(x))
               : "h"(__HALF_TO_US(x)));
  return x;

#else
  return half(cutlass::fast_tanh(float(x)));
#endif
#else
  return half(tanhf(float(x)));
#endif
}

__device__ bfloat16 fast_tanh(bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 900) && defined(AIT_USE_FAST_MATH)
  asm volatile("tanh.approx.bf16 %0, %1;"
               : "=h"(__HALF_TO_US(x))
               : "h"(__HALF_TO_US(x)));
  return x;

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#if defined(AIT_USE_FAST_MATH)
  return cutlass::fast_tanh(float(x));
#else
  return bfloat16(tanhf(float(x)));
#endif
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
// The Layernorm implementation below is based on OneFlow's Layernorm
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

template <typename SRC, typename DST>
struct TensorAccessorLoad {
  TensorAccessorLoad(
      const SRC* src,
      int64_t row_size,
      const TensorAccessor input_accessor)
      : src(src), row_size(row_size), input_accessor(input_accessor) {}

  template <int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    layer_norm::Pack<SRC, N> pack;
    pack.storage =
        *input_accessor.get<const SRC, const layer_norm::PackType<SRC, N>>(
            reinterpret_cast<const layer_norm::PackType<SRC, N>*>(src),
            (row * row_size + col) / N);

#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }

  bool CanPackAs(size_t pack_size) {
    return row_size % pack_size == 0 &&
        input_accessor.max_alignment() % pack_size == 0;
  }

  const SRC* src;
  int64_t row_size;
  const TensorAccessor input_accessor;
};

template <typename SRC, typename DST, bool FuseSigmoidMul>
struct TensorAccessorStore {
  TensorAccessorStore(
      DST* y,
      const DST* x,
      const DST* gamma,
      const DST* beta,
      int64_t row_size,
      const TensorAccessor input_accessor,
      const TensorAccessor output_accessor)
      : y(y),
        x(x),
        gamma(gamma),
        beta(beta),
        row_size(row_size),
        input_accessor(input_accessor),
        output_accessor(output_accessor) {}

  template <int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    layer_norm::Pack<DST, N> x_pack;
    layer_norm::Pack<DST, N> y_pack;

    if constexpr (FuseSigmoidMul) {
      x_pack.storage =
          *input_accessor.get<const DST, const layer_norm::PackType<DST, N>>(
              reinterpret_cast<const layer_norm::PackType<DST, N>*>(x),
              (row * row_size + col) / N);
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      SRC normalized_i = src[i];

#ifdef AIT_LAYERNORM_CONST_GAMMA
      const SRC gamma_val = AIT_LAYERNORM_CONST_GAMMA;
#else
      const SRC gamma_val = static_cast<SRC>(gamma[col + i]);
#endif // AIT_LAYERNORM_CONST_GAMMA

#ifdef AIT_LAYERNORM_CONST_BETA
      const SRC beta_val = AIT_LAYERNORM_CONST_BETA;
#else
      const SRC beta_val = static_cast<SRC>(beta[col + i]);
#endif // AIT_LAYERNORM_CONST_BETA

      normalized_i = normalized_i * gamma_val + beta_val;

      if constexpr (FuseSigmoidMul) {
        FSigmoid<SRC> fsigmoid;
        normalized_i =
            static_cast<SRC>(x_pack.elem[i]) * fsigmoid(normalized_i);
      }

      y_pack.elem[i] = DST(normalized_i);
    }

    *output_accessor.get<DST, layer_norm::PackType<DST, N>>(
        reinterpret_cast<layer_norm::PackType<DST, N>*>(y),
        (row * row_size + col) / N) = y_pack.storage;
  }

  bool CanPackAs(size_t pack_size) {
    return row_size % pack_size == 0 &&
        output_accessor.max_alignment() % pack_size == 0;
  }

  DST* y;
  const DST* x;
  const DST* gamma;
  const DST* beta;
  int64_t row_size;
  const TensorAccessor input_accessor;
  const TensorAccessor output_accessor;
};

template <typename TInput, typename TCompute, bool FuseSigmoidMul>
cudaError_t invokeLayernormSigmoidMul(
    TInput* output,
    const TInput* input,
    const TInput* gamma,
    const TInput* beta,
    int m,
    int n,
    const float eps,
    cudaStream_t stream,
    const TensorAccessor& input_accessor,
    const TensorAccessor& output_accessor) {
  TensorAccessorLoad<TInput, TCompute> load(input, n, input_accessor);
  TensorAccessorStore<TCompute, TInput, FuseSigmoidMul> store(
      output, input, gamma, beta, n, input_accessor, output_accessor);

  // mean and inv_variance are not required for forward pass, hence omitted
  layer_norm::DispatchLayerNorm<decltype(load), decltype(store), TCompute>(
      stream,
      load,
      store,
      m /* rows */,
      n /* cols */,
      eps /* epsilon */,
      nullptr /* mean */,
      nullptr /* inv_variance */);

  return cudaGetLastError();
}

#endif /* LAYERNORM_KERNEL_CUH */
