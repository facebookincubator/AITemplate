# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from . import common, common_bias_activation

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

namespace cutlass {
namespace epilogue {
namespace thread {

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
using LinearCombinationFastGELU = LinearCombinationGeneric<GELU_taylor, ElementOutput_, Count, ElementAccumulator_,
                                                          ElementCompute_, Scale, Round, true>;

} // namespace thread
} // namespace epilogue
} // namespace cutlass

"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) bias_ptr,
    (void*) (c_ptr + output_offset),
    M * K,
    N * K,
    N,
    M * N,
    K,
    K,
    0,
    output_stride
"""
)


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    dtype : str, optional
        [description], by default "float16"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    return common_bias_activation.gemm_rcr_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return common_bias_activation.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        PROBLEM_ARGS_TEMPLATE,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    return common_bias_activation.gen_function(
        func_attrs,
        PROBLEM_ARGS_TEMPLATE,
        exec_cond_template,
        dim_info_dict,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.func_decl")
def gen_function_decl(func_attrs):
    """[summary]

    Parameters
    ----------
    func_name : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return common_bias_activation.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.func_call")
def gen_function_call(func_attrs, indent="  "):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]
    """
    return common_bias_activation.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rcr_bias_fast_gelu.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
