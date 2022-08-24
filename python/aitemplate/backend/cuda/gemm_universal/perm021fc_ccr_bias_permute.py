# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Common functions and templates for bmm-family ops
"""
from ... import registry

from ..gemm_universal import common

from . import (
    bmm_common,
    bmm_permute_common,
    common_bias,
    common_permute,
    perm021fc_ccr_bias,
)


EXTRA_CODE = """

#include "cutlass/gemm/device/gemm_universal_with_perm.h"

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/coord.h"
#include "cutlass/tensor_coord.h"

namespace cutlass {
namespace layout {

template<int D0>
class Tensor3DPermute021BMM {
 public:
  using Index = int32_t;
  using LongIndex = int64_t;

  Index col_permute;
  Index row_permute;
  Index stride_permute;

 private:
  MatrixCoord extent_;

 public:
  CUTLASS_HOST_DEVICE
  Tensor3DPermute021BMM() {}

  CUTLASS_HOST_DEVICE
  Tensor3DPermute021BMM(MatrixCoord extent) : extent_(extent) {}

  CUTLASS_HOST_DEVICE
  void compute(Index col_init, Index row_init, Index stride_init, Index BMM_batch_idx) {
    // Permute as torch.permute(X1, [0, 2, 1]) -> 3D Tensor indices as [i,j,k], the dimension of X is [D0, D1, D2], after permutation the dim of X1 is [D0, D2, D1].
    // printf("BMM batch index: %d\t GEMM_m, GEMM_n = %d, %d\\n", BMM_batch_idx, extent_.row(), extent_.column());

    int k = col_init;
    int j = row_init;
    int i = BMM_batch_idx;

    col_permute = j;
    row_permute = k;
    stride_permute = stride_init / extent_.column() * extent_.row(); // stride in Bytes
  }
};

}  // namespace layout
}  // namespace cutlass
"""


@registry.reg("cuda.perm021fc_ccr_bias_permute.config")
def config(func_attrs, dtype="float16"):
    """
    [summary]
    """

    def fproc_f16(op):
        import cutlass_lib

        return common_permute.default_fproc_f16(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            epiligue_name=func_attrs["epilogue"],
            permute_layout=func_attrs["layout"],
        )

    func_attrs["op_instance"] = common_permute.extract_config(fproc_f16, func_attrs)


@registry.reg("cuda.perm021fc_ccr_bias_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    """
    [summary]
    """
    return perm021fc_ccr_bias.gen_profiler(func_attrs, workdir, dim_info_dict)


@registry.reg("cuda.perm021fc_ccr_bias_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = perm021fc_ccr_bias._get_problem_info(
        alpha_value=func_attrs.get("alpha", 1)
    )
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    return bmm_permute_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
        extra_code=EXTRA_CODE,
        has_bias=True,
    )


@registry.reg("cuda.perm021fc_ccr_bias_permute.func_decl")
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
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndims=input_ndims, weight_ndims=weight_ndims
    )


@registry.reg("cuda.perm021fc_ccr_bias_permute.func_call")
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
    bias = func_attrs["inputs"][2]
    return bmm_common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.perm021fc_ccr_bias_permute.filter")
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
