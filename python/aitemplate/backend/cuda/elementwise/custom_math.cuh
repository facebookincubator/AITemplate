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
#ifndef CUSTOM_MATH
#define CUSTOM_MATH

#ifndef __HALF2_TO_UI
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#endif

template <typename T>
__device__ T sign_custom(const T a) {
  return T(a > T(0)) - T(a < T(0));
}

__device__ half2 h2sign_custom(const half2 a) {
  return half2(sign_custom(a.x), sign_custom(a.y));
}

__device__ half2 fast_tanh(half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && \
    (__CUDA_ARCH__ >= 750)

  asm volatile("tanh.approx.f16x2 %0, %1;"
               : "=r"(__HALF2_TO_UI(x))
               : "r"(__HALF2_TO_UI(x)));
  return x;

#else
  CUTLASS_NOT_IMPLEMENTED();
#endif
}

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

// Return 1
__device__ half one() {
  uint16_t bits = 0x3c00u;
  return reinterpret_cast<half const&>(bits);
}

/// Returns (1/2)  (specialization for half_t)
__device__ half constant_half() {
  uint16_t bits = 0x3800u;
  return reinterpret_cast<half const&>(bits);
}

__device__ float fsigmoid_custom(const float a) {
  return (cutlass::fast_tanh(a * 0.5f) + 1.0f) * 0.5f;
}

__device__ half hsigmoid_custom(const half a) {
  half half_val = constant_half();
  half one_val = one();
  return __hmul((__hadd(fast_tanh(__hmul(a, half_val)), one_val)), half_val);
}

__device__ half2 h2sigmoid_custom(const half2 a) {
  half2 halfX2 = half2(constant_half(), constant_half());
  half2 oneX2 = half2(one(), one());
  return __hmul2((__hadd2(fast_tanh(__hmul2(a, halfX2)), oneX2)), halfX2);
}

__device__ float fsilu(const float a) {
  return a * fsigmoid_custom(a);
}

__device__ half hsilu(const half a) {
  return __hmul(a, hsigmoid_custom(a));
}

__device__ half2 h2silu(const half2 a) {
  return __hmul2(a, h2sigmoid_custom(a));
}

__device__ float leaky_relu(const float a, const float negativeSlope) {
  return a > 0.f ? a : a * negativeSlope;
}

__device__ half leaky_relu(const half a, const half negativeSlope) {
  return a > half(0.f) ? a : __hmul(a, negativeSlope);
}

__device__ half2 leaky_relu(const half2 a, const half2 negativeSlope) {
  return half2(
      leaky_relu(a.x, negativeSlope.x), leaky_relu(a.y, negativeSlope.y));
}

__device__ float relu(const float a) {
  return a > 0.f ? a : 0.f;
}

__device__ half relu(const half a) {
  return a > half(0.f) ? a : half(0.f);
}

__device__ half2 relu(const half2 a) {
  half2 zeroX2 = half2(half(0.f), half(0.f));
#if __CUDA_ARCH__ >= 800
  return __hmax2(a, zeroX2);
#else
  return half2(relu(a.x), relu(a.y));
#endif
}

template <typename T>
__device__ T hard_tanh(const T a, T min_val, T max_val) {
  if (a <= min_val) {
    return min_val;
  } else if (a >= max_val) {
    return max_val;
  } else {
    return a;
  }
}

__device__ half2
h2hard_tanh(const half2 a, const half2 min_val, const half2 max_val) {
  return half2(
      hard_tanh(a.x, min_val.x, max_val.x),
      hard_tanh(a.y, min_val.y, max_val.y));
}

__device__ half replace_if_inf(
    const half a,
    const half inf_replace,
    const half neginf_replace) {
  auto is_inf = __hisinf(a);
  if (is_inf == -1) {
    return neginf_replace;
  }
  if (is_inf == 1) {
    return inf_replace;
  }
  return a;
}

__device__ float replace_if_inf(
    const float a,
    const float inf_replace,
    const float neginf_replace) {
  auto is_inf = isinf(a);
  if (is_inf == -1) {
    return neginf_replace;
  }
  if (is_inf == 1) {
    return inf_replace;
  }
  return a;
}

__device__ half2 nan_to_num(
    const half2 a,
    const half2 nan_replace,
    const half2 inf_replace,
    const half2 neginf_replace) {
  half2 isnan = __hisnan2(a);
  return half2(
      isnan.x ? nan_replace.x
              : replace_if_inf(a.x, inf_replace.x, neginf_replace.x),
      isnan.y ? nan_replace.y
              : replace_if_inf(a.y, inf_replace.y, neginf_replace.y));
}

__device__ half nan_to_num(
    const half a,
    const half nan_replace,
    const half inf_replace,
    const half neginf_replace) {
  if (__hisnan(a)) {
    return nan_replace;
  }
  return replace_if_inf(a, inf_replace, neginf_replace);
}

__device__ float nan_to_num(
    const float a,
    const float nan_replace,
    const float inf_replace,
    const float neginf_replace) {
  if (isnan(a)) {
    return nan_replace;
  }
  return replace_if_inf(a, inf_replace, neginf_replace);
}

__device__ half2 clamp_nan_to_num(
    const half2 a,
    const half2 clamp_min,
    const half2 clamp_max,
    const half2 nan_replace) {
  half2 isnan = __hisnan2(a);
  return half2(
      isnan.x ? nan_replace.x : hard_tanh(a.x, clamp_min.x, clamp_max.x),
      isnan.y ? nan_replace.y : hard_tanh(a.y, clamp_min.y, clamp_max.y));
}

__device__ half clamp_nan_to_num(
    const half a,
    const half clamp_min,
    const half clamp_max,
    const half nan_replace) {
  return __hisnan(a) ? nan_replace : hard_tanh(a, clamp_min, clamp_max);
}

__device__ float clamp_nan_to_num(
    const float a,
    const float clamp_min,
    const float clamp_max,
    const float nan_replace) {
  return isnan(a) ? nan_replace : hard_tanh(a, clamp_min, clamp_max);
}

// Backup functions for CUDA_ARCH < 800
__device__ half nanh() {
  return __float2half(nanf(""));
}

__device__ bool half_isnan(half h) {
  return h != h;
}

__device__ half hmin(half a, half b) {
  return (a < b) ? a : b;
}

__device__ half hmax(half a, half b) {
  return (a > b) ? a : b;
}

// max/min functions that let NaNs pass through
__device__ float fmaxf_nan(const float a, const float b) {
  return (isnan(a) || isnan(b)) ? nanf("") : fmaxf(a, b);
}

__device__ half hmax_nan(const half a, const half b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmax_nan(a, b);
#else
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmax(a, b);
#endif
}

__device__ half2 hmax2_nan(const half2 a, const half2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmax2_nan(a, b);
#else
  return half2(hmax_nan(a.x, b.x), hmax_nan(a.y, b.y));
#endif
}

__device__ float fminf_nan(const float a, const float b) {
  return (isnan(a) || isnan(b)) ? nanf("") : fminf(a, b);
}

__device__ half hmin_nan(const half a, const half b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmin_nan(a, b);
#else
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmin(a, b);
#endif
}

__device__ half2 hmin2_nan(const half2 a, const half2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hmin2_nan(a, b);
#else
  return half2(hmin_nan(a.x, b.x), hmin_nan(a.y, b.y));
#endif
}

// pow impl
__device__ half hpow(const half a, const half b);

__device__ half2 h2pow(const half2 a, const half2 b) {
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  if (b1 != b2) {
    half a1 = __low2half(a);
    half a2 = __high2half(a);
    half c1 = hpow(a1, b1);
    half c2 = hpow(a2, b2);
    return __halves2half2(c1, c2);
  }

  // New special cases can be added if needed, such as
  // an powi for cases where b is an integer
  if (__hbeq2(b, half2(0.0, 0.0))) {
    return half2(1.0, 1.0);
  }
  if (__hbeq2(b, half2(1.0, 1.0))) {
    return a;
  }
  if (__hbeq2(b, half2(2.0, 2.0))) {
    return __hmul2(a, a);
  }
  if (__hbeq2(b, half2(3.0, 3.0))) {
    return __hmul2(__hmul2(a, a), a);
  }
  if (__hbeq2(b, half2(0.5, 0.5))) {
    return h2sqrt(a);
  }
  if (__hbeq2(b, half2(-0.5, -0.5))) {
    return h2rsqrt(a);
  }
  if (__hbeq2(b, half2(-1.0, -1.0))) {
    return __h2div(half2(1.0, 1.0), a);
  }
  if (__hbeq2(b, half2(-2.0, -2.0))) {
    return __h2div(half2(1.0, 1.0), __hmul2(a, a));
  }

  half a1 = __low2half(a);
  half a2 = __high2half(a);

  // low 16 bits
  half c1 =
      static_cast<half>(pow(static_cast<double>(a1), static_cast<double>(b1)));
  // high 16 bits
  half c2 =
      static_cast<half>(pow(static_cast<double>(a2), static_cast<double>(b2)));
  return __halves2half2(c1, c2);
}

__device__ half hpow(const half a, const half b) {
  if (b == half(0.0)) {
    return half(1.0);
  }
  if (b == half(1.0)) {
    return a;
  }
  if (b == half(2.0)) {
    return a * a;
  }
  if (b == half(3.0)) {
    return a * a * a;
  }
  if (b == half(0.5)) {
    return hsqrt(a);
  }
  if (b == half(-0.5)) {
    return hrsqrt(a);
  }
  if (b == half(-1.0)) {
    return half(1.0) / a;
  }
  if (b == half(-2.0)) {
    return half(1.0) / (a * a);
  }
  return static_cast<half>(pow(static_cast<double>(a), static_cast<double>(b)));
}

__device__ float fpow(const float a, const float b) {
  if (b == float(0.0)) {
    return float(1.0);
  }
  if (b == float(1.0)) {
    return a;
  }
  if (b == float(2.0)) {
    return a * a;
  }
  if (b == float(3.0)) {
    return a * a * a;
  }
  if (b == float(0.5)) {
    return sqrt(a);
  }
  if (b == float(-0.5)) {
    return rsqrt(a);
  }
  if (b == float(-1.0)) {
    return float(1.0) / a;
  }
  if (b == float(-2.0)) {
    return float(1.0) / (a * a);
  }
  return static_cast<float>(
      pow(static_cast<double>(a), static_cast<double>(b)));
}

//
// GELU function definitions implemented as described by
//   Hendrycks, D., and Gimpel, K. in
//   "Gaussian Error Linear Units (GELUs)." (2020)
//   https://arxiv.org/pdf/1606.08415.pdf
//
// Floating-point constants are Taylor coefficients described in the paper.
//
__device__ half hgelu(const half a) {
  cutlass::epilogue::thread::GELU<cutlass::half_t> gelu_op;
  return static_cast<half>(gelu_op(static_cast<cutlass::half_t>(a)));
}

__device__ float fgelu(const float a) {
  cutlass::epilogue::thread::GELU<float> gelu_op;
  return gelu_op(a);
}

__device__ half h_fast_gelu(const half a) {
  cutlass::epilogue::thread::GELU_taylor<cutlass::half_t> gelu_op;
  return static_cast<half>(gelu_op(static_cast<cutlass::half_t>(a)));
}

__device__ float f_fast_gelu(const float a) {
  cutlass::epilogue::thread::GELU_taylor<float> gelu_op;
  return gelu_op(a);
}

__device__ float fsoftplus(
    const float a,
    const float beta,
    const float threshold) {
  return (a * beta > threshold) ? a : log1pf(expf(a * beta)) / beta;
}

__device__ half hsoftplus(const half a, const half beta, const half threshold) {
  half one_val = one();
  return __hgt(__hmul(a, beta), threshold)
      ? a
      : __hdiv(hlog(__hadd(one_val, hexp(__hmul(a, beta)))), beta);
}

__device__ half2
h2softplus(const half2 a, const half2 beta, const half2 threshold) {
  return half2(
      hsoftplus(a.x, beta.x, threshold.x), hsoftplus(a.y, beta.y, threshold.y));
}

#endif
