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
GEMM Specialization for C = fast_gelu(GeMM(A, B) + bias)
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][K], C[RowMajor][M, N]
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import common, common_bias_activation

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    {M, N, K},                                               // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr),                           // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr),                           // void const * ptr_B
    ({{elem_input_type}}*)(bias_ptr),                        // void const * ptr_C
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    N,                                                       // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    K,                                                       // typename LayoutB::Stride::LongIndex ldb
    0,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


@registry.reg("cuda.gemm_rcr_bias_gelu.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    return common_bias_activation.gemm_rcr_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias_gelu.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return common_bias_activation.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_bias_gelu.gen_function")
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
    )


@registry.reg("cuda.gemm_rcr_bias_gelu.func_decl")
def gen_function_decl(func_attrs):
    return common_bias_activation.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_gelu.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common_bias_activation.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rcr_bias_gelu.filter")
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
