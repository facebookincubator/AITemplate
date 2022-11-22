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
C = RELU(ADD(ADD(GeMM(A, B) + bias, D0), D1))
where A[RowMajor][M, K], B[ColMajor][N, K], C[RowMajor][M, N]
bias[RowMajor][N], D0[RowMajor][M, N], D1[RowMajor][M, N]
"""
from ... import registry
from . import common, common_bias_broadcast
from .layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

UNARY_OP1 = "cutlass::epilogue::thread::Identity"
BINARY_OP1 = "cutlass::plus"
BINARY_OP2 = "cutlass::plus"
UNARY_OP2 = "cutlass::epilogue::thread::Identity"


@registry.reg("cuda.gemm_rcr_bias_add_add.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    return common_bias_broadcast.gemm_bias_broadcast_config(func_attrs, RCR)


@registry.reg("cuda.gemm_rcr_bias_add_add.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return common_bias_broadcast.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        RCR,
        UNARY_OP1,
        BINARY_OP1,
        BINARY_OP2,
        UNARY_OP2,
    )


@registry.reg("cuda.gemm_rcr_bias_add_add.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    return common_bias_broadcast.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        RCR,
        UNARY_OP1,
        BINARY_OP1,
        BINARY_OP2,
        UNARY_OP2,
    )


@registry.reg("cuda.gemm_rcr_bias_add_add.func_decl")
def gen_function_decl(func_attrs):
    return common_bias_broadcast.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_add_add.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common_bias_broadcast.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rcr_bias_add_add.filter")
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
