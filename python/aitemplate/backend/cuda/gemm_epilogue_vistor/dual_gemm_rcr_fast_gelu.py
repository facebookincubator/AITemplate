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
C = FAST_GELU(GEMM_RCR(A, B)) * GEMM_RCR(A, B1)
where A[RowMajor][M, K], B[ColMajor][N, K], B1[RowMajor][N, K]
"""
import jinja2

from ... import registry
from ...backend_spec import CUDASpec
from ..gemm_universal import common, common_bias
from ..gemm_universal.layout import RCR
from . import common_dual_gemm

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmCoord{M, N, K},
    {({{elem_input_type}}*)a_ptr, LayoutA(K)},
    {({{elem_input_type}}*)b_ptr, LayoutB(K)},
    ref_B0,
    nullptr_ref, // D0
    {({{elem_input_type}}*)bias_ptr, LayoutB(K)}, // B1
    ref_B1,
    nullptr_ref, // D1
    {({{elem_output_type}}*)c_ptr, LayoutC(N)}, // D2
    {ElementCompute(1), ElementCompute(0)},
    {ElementCompute(1), ElementCompute(0)},
    {},
    1 // kSplitKSerial
"""
)


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
    int64_t M = std::atoi(argv[1]);
    int64_t N = std::atoi(argv[2]);
    int64_t K = std::atoi(argv[3]);
    int64_t split_k = std::atoi(argv[4]);

    int64_t a_dim0 = M;
    int64_t a_dim1 = K;
    int64_t b_dim0 = N;
    int64_t b_dim1 = K;
    int64_t c_dim0 = M;
    int64_t c_dim1 = N;
"""
)

# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmCoord{M, N, K},
    {({{elem_input_type}}*)a_ptr, LayoutA(K)},
    {({{elem_input_type}}*)b_ptr, LayoutB(K)},
    ref_B0,
    nullptr_ref, // D0
    {({{elem_input_type}}*)bias_ptr, LayoutB(K)}, // B1
    ref_B1,
    nullptr_ref, // D1
    {({{elem_output_type}}*)c_ptr, LayoutC(N)}, // D2
    {ElementCompute(1), ElementCompute(0)},
    {ElementCompute(1), ElementCompute(0)},
    {},
    1 // kSplitKSerial
"""
)


EXTRA_CODE = jinja2.Template(
    """
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "device/dual_gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation.
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LeftFastGeluAndMul {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params{};

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LeftFastGeluAndMul(Params const &/*params*/) {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    assert(false);
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &lhs,
    FragmentAccumulator const &rhs) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_to_compute;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> compute_to_output;

    ComputeFragment converted_lhs = accumulator_to_compute(lhs);
    ComputeFragment converted_rhs = accumulator_to_compute(rhs);

    cutlass::epilogue::thread::GELU_taylor<ComputeFragment> gelu;
    cutlass::multiplies<ComputeFragment> mul;
    auto gelu_lhs = gelu(converted_lhs);
    return compute_to_output(mul(gelu_lhs, converted_rhs));
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
      ElementAccumulator const& lhs,
      ElementAccumulator const& rhs
  ) const {
      ElementCompute convert_lhs(lhs);
      ElementCompute convert_rhs(rhs);
      cutlass::epilogue::thread::GELU_taylor<ElementCompute> gelu;
      cutlass::multiplies<ElementCompute> mul;
      auto gelu_lhs = gelu(convert_lhs);
      return ElementOutput(mul(gelu_lhs, convert_rhs));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass


typename cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> nullptr_ref{};
decltype(nullptr_ref) ref_B0, ref_B1;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

"""
)


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    common_dual_gemm.make_fproc_f16(func_attrs, RCR)


def common_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    bias_ptr_arg=None,
    extra_code="",
):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )
    return common_dual_gemm.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        src_template,
        problem_args_template,
        ARGS_PARSER_TEMPLATE,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
        bias_ptr_arg=bias_ptr_arg,
        extra_code=extra_code,
    )


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return common_gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common_bias.SRC_TEMPLATE,
        PROFILER_PROBLEM_ARGS_TEMPLATE,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    if problem_args_template is None:
        problem_args = PROBLEM_ARGS_TEMPLATE.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    else:
        problem_args = problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    return common_dual_gemm.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        output_ndims,
        dim_info_dict,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.dual_gemm_rcr_fast_gelu.filter")
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
