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
argmax kernel codegen for CUDA.
"""

from typing import Any, Dict

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.tensor import argmax_common

# pylint: disable=C0301

header_files = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
#include <type_traits>

using bfloat16 = nv_bfloat16;

namespace cub {

namespace {

template <typename... Ts>
using void_t = void;

template <typename T, typename = void>
struct is_defined : std::false_type {};

// We rely on the fact that defined classes can be instantiated, and the size
// of their instance can be calculated. Therefore, if sizeof() fails, then
// SFINAE will fall back to previous option (i.e. std::false_type).
template <typename T>
struct is_defined<T, void_t<decltype(sizeof(T))>> : std::true_type {};

}  // namespace

struct ThrowAwayType {}; // don't care about this template type

// A forward declaration is needed in case this type doesn't exist yet.
template <> struct FpLimits<bfloat16>;

using FpLimitsType = std::conditional_t<is_defined<FpLimits<bfloat16>>::value, ThrowAwayType, bfloat16>;

template <>
struct FpLimits<FpLimitsType>
{
    static __host__ __device__ __forceinline__ bfloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<bfloat16&>(max_word);
    }

    static __host__ __device__ __forceinline__ bfloat16 Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<bfloat16&>(lowest_word);
    }
};

// A forward declaration is needed in case this type doesn't exist yet.
template <> struct NumericTraits<bfloat16>;

using NumericTraitsType = std::conditional_t<is_defined<NumericTraits<bfloat16>>::value, ThrowAwayType, bfloat16>;

template <> struct NumericTraits<NumericTraitsType>
  : BaseTraits<FLOATING_POINT, true, false, unsigned short, bfloat16> {};

template<> struct Traits<NumericTraitsType>
  : NumericTraits<bfloat16> {};

}  // namespace cub

"""


@registry.reg("cuda.argmax.gen_function")
def argmax_gen_function(func_attrs: Dict[str, Any]) -> str:
    return argmax_common.gen_function(func_attrs, header_files, CUDASpec())


@registry.reg("cuda.argmax.func_decl")
def argmax_gen_function_decl(func_attrs: Dict[str, Any]):
    return argmax_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.argmax.func_call")
def argmax_gen_function_call(func_attrs, indent="  "):
    return argmax_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.argmax.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return argmax_common.gen_profiler(func_attrs, workdir, header_files, CUDASpec())
