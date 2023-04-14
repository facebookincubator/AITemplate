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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_epilogue_vistor import (
    common_softmax,
    gemm_rcr_softmax,
)
from aitemplate.backend.cuda.gemm_universal import common
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);

  int64_t a_dim0 = B;
  int64_t a_dim1 = M;
  int64_t a_dim2 = K;
  int64_t b_dim0 = B;
  int64_t b_dim1 = N;
  int64_t b_dim2 = K;
  int64_t c_dim0 = B;
  int64_t c_dim1 = M;
  int64_t c_dim2 = N;
"""
)

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    /*
        A: (B, M, K) (RowMajor)
        B: (B, N, K) (ColumnMajor)
        C, D, Soft: (B, M, N) (RowMajor)
        N, S: (B, block_num, M) (RowMajor)
    */

    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                                                                                                     // cutlass::gemm::GemmCoord problem_size
    B,                                                                                                                                     // int32_t batch_count_
    {reinterpret_cast<{{elem_input_type}}*>(a_ptr), LayoutA(K)},                                                                           // TensorRefA ref_A_
    {reinterpret_cast<{{elem_input_type}}*>(b_ptr), LayoutB(K)},                                                                           // TensorRefB ref_B_
    {reinterpret_cast<{{elem_output_type}}*>(workspace), LayoutC(N)},                                                                      // TensorRefC ref_C_
    {reinterpret_cast<{{elem_output_type}}*>(workspace + B * M * N * sizeof({{elem_output_type}})), LayoutC(N)},                           // TensorRefC ref_D_
    {
        float(1.0),
        float(0.0)
    },                                                                                                                                     // typename EpilogueFunctorOp::Params linear_scaling
    {reinterpret_cast<float*>(workspace + 2 * B * M * N * sizeof({{elem_output_type}})), LayoutC(1)},                                      // TensorRefN ref_N_
    {reinterpret_cast<float*>(workspace + 2 * B * M * N * sizeof({{elem_output_type}}) + B * M * block_num * sizeof(float)), LayoutC(1)},  // TensorRefSum ref_S_
    {reinterpret_cast<{{elem_output_type}}*>(soft_ptr) + output_offset, LayoutC(output_stride)},                                           // TensorRefSoft ref_Softmax_
    M * K,                                                                                                                                 // int64_t batch_stride_A_
    N * K,                                                                                                                                 // int64_t batch_stride_B_
    M * N,                                                                                                                                 // int64_t batch_stride_C_
    M * N,                                                                                                                                 // int64_t batch_stride_D_
    M * block_num,                                                                                                                         // int64_t batch_stride_Max_
    M * block_num,                                                                                                                         // int64_t batch_stride_Sum_
    M * N                                                                                                                                  // int64_t batch_stride_Softmax_
"""
)


@registry.reg("cuda.bmm_rcr_softmax.config")
def bmm_rcr_softmax_config(func_attrs, dtype="float16"):
    common.make_fproc(func_attrs, RCR)


@registry.reg("cuda.bmm_rcr_softmax.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
):
    return gemm_rcr_softmax.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_softmax.SRC_TEMPLATE,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        args_parser_template=ARGS_PARSER_TEMPLATE,
        ndims=3,
    )


@registry.reg("cuda.bmm_rcr_softmax.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    return gemm_rcr_softmax.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.bmm_rcr_softmax.func_decl")
def gen_function_decl(func_attrs):
    return gemm_rcr_softmax.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("cuda.bmm_rcr_softmax.func_call")
def gen_function_call(func_attrs, indent="  "):
    return gemm_rcr_softmax.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("cuda.bmm_rcr_softmax.filter")
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
