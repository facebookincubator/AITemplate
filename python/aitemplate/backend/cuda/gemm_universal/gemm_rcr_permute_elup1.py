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
GEMM Specialization for
C = permute(elu(GeMM(A, B) + bias) + 1.0)
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][N]
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import gemm_rcr_permute

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


EXTRA_CODE = jinja2.Template(
    """
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

#define CUDA_FP16_ZERO \
  __half {             \
    0x0u               \
  }

#define CUDA_FP16_ONE \
  __half_raw {        \
    0x3c00u           \
  }

namespace cutlass {
namespace epilogue {
namespace thread {

// ELU(x; alpha = 1) + 1
template <typename T>
struct ELUp1 {
  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar) const {
    return scalar >= T(0) ? scalar + T(1) : fast_exp(scalar);
  }

  using Params = LinearCombinationGenericParams<T>;

  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar, Params const& params_) const {
    return this->operator()(scalar);
  }
};

template <>
struct ELUp1<cutlass::half_t> {
  CUTLASS_DEVICE
  cutlass::half_t operator()(cutlass::half_t const& scalar) const {
    half s = (half)scalar;
    return (cutlass::half_t)(
        __hadd(
            __hmul(__hgt(s, CUDA_FP16_ZERO), __hadd(s, CUDA_FP16_ONE)),
            __hmul(__hle(s, CUDA_FP16_ZERO), hexp(s))
        )
    );
  }

  using Params = LinearCombinationGenericParams<cutlass::half_t>;

  CUTLASS_DEVICE
  cutlass::half_t operator()(cutlass::half_t const& scalar, Params const& params_) const {
    return this->operator()(scalar);
  }
};

template <typename T, int N>
struct ELUp1<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& value) const {
    Array<T, N> y;
    ELUp1<T> elup1_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = elup1_op(value[i]);
    }

    return y;
  }

  using Params = LinearCombinationGenericParams<T>;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& value, Params const& params_)
      const {
    return this->operator()(value);
  }
};

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationELUp1 = LinearCombinationGeneric<ELUp1, ElementOutput_, Count, ElementAccumulator_,
                                                          ElementCompute_, Scale, Round, false>;

// The last template argument above (IsHeavy) being "false" is important for the functor
// (here: ELUp1) to be inlined. Otherwise, the performance of the epilogue may worsen.
// https://github.com/NVIDIA/cutlass/blob/7bdba07310b497e75c8377031e524fadc929b849/include/cutlass/epilogue/threadblock/epilogue_base.h#L74-L81

} // namespace thread
} // namespace epilogue
} // namespace cutlass

"""
)


@registry.reg("cuda.gemm_rcr_permute_elup1.config")
def gemm_rcr_permute_elup1_config(
    func_attrs,
    dtype="float16",
):
    gemm_rcr_permute.gemm_rcr_permute_config(
        func_attrs=func_attrs,
        dtype=dtype,
    )


@registry.reg("cuda.gemm_rcr_permute_elup1.gen_profiler")
def gemm_rcr_permute_elup1_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
):
    return gemm_rcr_permute.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_permute_elup1.gen_function")
def gemm_rcr_permute_elup1_gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
):
    return gemm_rcr_permute.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
        problem_args_template=problem_args_template,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_permute_elup1.func_decl")
def gemm_rcr_permute_elup1_func_decl(func_attrs):
    return gemm_rcr_permute.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_permute_elup1.func_call")
def gemm_rcr_permute_elup1_func_call(
    func_attrs,
    indent="  ",
):
    return gemm_rcr_permute.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("cuda.gemm_rcr_permute_elup1.filter")
def gemm_rcr_permute_elup1_filter(
    cfg,
    func_attrs,
    ab_alignment,
):
    return gemm_rcr_permute.function_filter(
        cfg=cfg,
        func_attrs=func_attrs,
        ab_alignment=ab_alignment,
    )
