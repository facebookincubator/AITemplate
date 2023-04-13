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


# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    /*
        A: (M, K) (RowMajor)
        B: (N, K) (ColumnMajor)
        C, D, Soft: (M, N) (RowMajor)
        N, S: (block_num, M) (RowMajor)
    */

    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                                                                                             // cutlass::gemm::GemmCoord problem_size
    1,                                                                                                                             // int32_t batch_count_
    {reinterpret_cast<{{elem_input_type}}*>(a_ptr), LayoutA(K)},                                                                   // TensorRefA ref_A_
    {reinterpret_cast<{{elem_input_type}}*>(b_ptr), LayoutB(K)},                                                                   // TensorRefB ref_B_
    {reinterpret_cast<{{elem_output_type}}*>(bias_ptr), 0},                                                                        // TensorRefC ref_C_
    {reinterpret_cast<{{elem_output_type}}*>(workspace + M * N * sizeof({{elem_output_type}})), LayoutC(N)},                       // TensorRefC ref_D_
    {
        float(1.0),
        float(1.0)
    },                                                                                                                             // typename EpilogueFunctorOp::Params linear_scaling
    {reinterpret_cast<float*>(workspace + 2 * M * N * sizeof({{elem_output_type}})), LayoutC(1)},                                  // TensorRefN ref_N_
    {reinterpret_cast<float*>(workspace + 2 * M * N * sizeof({{elem_output_type}}) + M * block_num * sizeof(float)), LayoutC(1)},  // TensorRefSum ref_S_
    {reinterpret_cast<{{elem_output_type}}*>(soft_ptr) + output_offset, LayoutC(output_stride)},                                   // TensorRefSoft ref_Softmax_
"""
)


@registry.reg("cuda.gemm_rcr_bias_softmax.config")
def gemm_rcr_bias_softmax_config(func_attrs, dtype="float16"):
    gemm_rcr_softmax.gemm_rcr_softmax_config(
        func_attrs=func_attrs,
        dtype=dtype,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.gen_profiler")
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
        args_parser_template=gemm_rcr_softmax.ARGS_PARSER_TEMPLATE,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.gen_function")
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
        has_bias=True,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.func_decl")
def gen_function_decl(func_attrs):
    return gemm_rcr_softmax.gen_function_decl(
        func_attrs=func_attrs,
        has_bias=True,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]

    return gemm_rcr_softmax.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        has_bias=True,
        bias_ptr=bias._attrs["name"],
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.filter")
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
