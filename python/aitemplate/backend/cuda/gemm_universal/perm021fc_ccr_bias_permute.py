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
Common functions and templates for perm021_ccr_bias_permute, which computes
(A.permute(0, 2, 1)[col] @ B[col] + Bias).permute(0, 2, 1)
"""
from aitemplate.backend import registry

from aitemplate.backend.cuda.gemm_universal import (
    bmm_common,
    bmm_permute_common,
    common,
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
    def fproc(op):
        import cutlass_lib

        return common.default_fproc(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            dtype=func_attrs["inputs"][0].dtype(),
            epilogue_name=func_attrs["epilogue"],
            permute_layout=func_attrs["layout"],
        )

    func_attrs["op_instance"] = common_permute.extract_config(fproc, func_attrs)


@registry.reg("cuda.perm021fc_ccr_bias_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return perm021fc_ccr_bias.gen_profiler(
        func_attrs, workdir, profiler_filename, dim_info_dict
    )


@registry.reg("cuda.perm021fc_ccr_bias_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = perm021fc_ccr_bias._get_problem_info(
        alpha_value=func_attrs.get("alpha", 1),
    )
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=mm_info,
    )

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
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndims=input_ndims, weight_ndims=weight_ndims
    )


@registry.reg("cuda.perm021fc_ccr_bias_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
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
