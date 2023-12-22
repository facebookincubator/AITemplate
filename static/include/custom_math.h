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

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

using bfloat16 = hip_bfloat16;


#include <hip/math_functions.h>
#include <hip/device_functions.h>


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
  return half2(
      sign_custom(*reinterpret_cast<const half*>(&a.x)),
      sign_custom(*reinterpret_cast<const half*>(&a.y)));
}

__device__ half2 fast_tanh(half2 x) {
  // 1-2/(e^(2x)+1)
  const half2 u = __hmul2(half2(2), x);
  const half2 emu = h2exp(u);
  const half2 cdf =
      __hsub2(half2(1), __h2div(half2(2), __hadd2(half2(1), emu)));
  return cdf;
}

__device__ half fast_tanh(half x) {
  // 1-2/(e^(2x)+1)
  const half u = __hmul(half(2), x);
  const half emu = hexp(u);
  const half cdf = __hsub(half(1), __hdiv(half(2), __hadd(half(1), emu)));
  return cdf;
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
  return (tanhf(a * 0.5f) + 1.0f) * 0.5f;
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

__device__ half hsin_custom(const half a) {
  float x = __half2float(a);
  return __float2half_rn(sin(x));
}

__device__ float leaky_relu(const float a, const float negativeSlope) {
  return a > 0.f ? a : a * negativeSlope;
}

__device__ half leaky_relu(const half a, const half negativeSlope) {
  return a > half(0.f) ? a : __hmul(a, negativeSlope);
}

__device__ half2 leaky_relu(const half2 a, const half2 negativeSlope) {
  return half2(
      leaky_relu(
          *reinterpret_cast<const half*>(&a.x),
          *reinterpret_cast<const half*>(&negativeSlope.x)),
      leaky_relu(
          *reinterpret_cast<const half*>(&a.y),
          *reinterpret_cast<const half*>(&negativeSlope.y)));
}

__device__ float relu(const float a) {
  return a > 0.f ? a : 0.f;
}

__device__ half relu(const half a) {
  return a > half(0.f) ? a : half(0.f);
}

__device__ half2 relu(const half2 a) {
  half2 zeroX2 = half2(half(0.f), half(0.f));
  return half2(
      relu(*reinterpret_cast<const half*>(&a.x)),
      relu(*reinterpret_cast<const half*>(&a.y)));
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
      hard_tanh(
          *reinterpret_cast<const half*>(&a.x),
          *reinterpret_cast<const half*>(&min_val.x),
          *reinterpret_cast<const half*>(&max_val.x)),
      hard_tanh(
          *reinterpret_cast<const half*>(&a.y),
          *reinterpret_cast<const half*>(&min_val.y),
          *reinterpret_cast<const half*>(&max_val.y)));
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
      *reinterpret_cast<const half*>(&isnan.x)
          ? *reinterpret_cast<const half*>(&nan_replace.x)
          : replace_if_inf(
                *reinterpret_cast<const half*>(&a.x),
                *reinterpret_cast<const half*>(&inf_replace.x),
                *reinterpret_cast<const half*>(&neginf_replace.x)),
      *reinterpret_cast<const half*>(&isnan.y)
          ? *reinterpret_cast<const half*>(&nan_replace.y)
          : replace_if_inf(
                *reinterpret_cast<const half*>(&a.y),
                *reinterpret_cast<const half*>(&inf_replace.y),
                *reinterpret_cast<const half*>(&neginf_replace.y)));
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
      *reinterpret_cast<const half*>(&isnan.x)
          ? *reinterpret_cast<const half*>(&nan_replace.x)
          : hard_tanh(
                *reinterpret_cast<const half*>(&a.x),
                *reinterpret_cast<const half*>(&clamp_min.x),
                *reinterpret_cast<const half*>(&clamp_max.x)),
      *reinterpret_cast<const half*>(&isnan.y)
          ? *reinterpret_cast<const half*>(&nan_replace.y)
          : hard_tanh(
                *reinterpret_cast<const half*>(&a.y),
                *reinterpret_cast<const half*>(&clamp_min.y),
                *reinterpret_cast<const half*>(&clamp_max.y)));
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

// Backup functions
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
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmax(a, b);
}

__device__ half2 hmax2_nan(const half2 a, const half2 b) {
  return half2(
      hmax_nan(
          *reinterpret_cast<const half*>(&a.x),
          *reinterpret_cast<const half*>(&b.x)),
      hmax_nan(
          *reinterpret_cast<const half*>(&a.y),
          *reinterpret_cast<const half*>(&b.y)));
}

__device__ float fminf_nan(const float a, const float b) {
  return (isnan(a) || isnan(b)) ? nanf("") : fminf(a, b);
}

__device__ half hmin_nan(const half a, const half b) {
  return (half_isnan(a) || half_isnan(b)) ? nanh() : hmin(a, b);
}

__device__ half2 hmin2_nan(const half2 a, const half2 b) {
  return half2(
      hmin_nan(
          *reinterpret_cast<const half*>(&a.x),
          *reinterpret_cast<const half*>(&b.x)),
      hmin_nan(
          *reinterpret_cast<const half*>(&a.y),
          *reinterpret_cast<const half*>(&b.y)));
}

#endif
