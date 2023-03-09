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
C = UnaryOp2(BinaryOp2(BinaryOp1(UnaryOp1(GeMM(A, B) + bias), D1), D2)),
"""
from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import common, common_bias_broadcast
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

UNARY_IDENTITY = "cutlass::epilogue::thread::Identity"
UNARY_RELU = "cutlass::epilogue::thread::ReLu"
UNARY_SIGMOID = "cutlass::epilogue::thread::Sigmoid"
UNARY_TANH = "cutlass::epilogue::thread::Tanh"
BINARY_PLUS = "cutlass::plus"
BINARY_MULTIPLY = "cutlass::multiplies"


_CONFIGS = [
    # gemm_rcr_bias_add
    ["add", UNARY_IDENTITY, BINARY_PLUS, None, UNARY_IDENTITY],
    # gemm_rcr_bias_add_add
    ["add_add", UNARY_IDENTITY, BINARY_PLUS, BINARY_PLUS, UNARY_IDENTITY],
    # gemm_rcr_bias_add_relu
    ["add_relu", UNARY_IDENTITY, BINARY_PLUS, None, UNARY_RELU],
    # gemm_rcr_bias_add_add_relu
    ["add_add_relu", UNARY_IDENTITY, BINARY_PLUS, BINARY_PLUS, UNARY_RELU],
    # gemm_rcr_bias_mul
    ["mul", UNARY_IDENTITY, BINARY_MULTIPLY, None, UNARY_IDENTITY],
    # gemm_rcr_bias_mul_add
    ["mul_add", UNARY_IDENTITY, BINARY_MULTIPLY, BINARY_PLUS, UNARY_IDENTITY],
    # gemm_rcr_bias_mul_tanh
    ["mul_tanh", UNARY_IDENTITY, BINARY_MULTIPLY, None, UNARY_TANH],
    # gemm_rcr_bias_sigmoid_mul_tanh
    ["sigmoid_mul_tanh", UNARY_SIGMOID, BINARY_MULTIPLY, None, UNARY_TANH],
    # gemm_rcr_bias_sigmoid_mul
    ["sigmoid_mul", UNARY_SIGMOID, BINARY_MULTIPLY, None, UNARY_IDENTITY],
]


def gemm_rcr_config(func_attrs, dtype="float16"):
    return common_bias_broadcast.gemm_bias_broadcast_config(func_attrs, RCR)


def gen_profiler_template(unary_op1, binary_op1, binary_op2, unary_op2):
    def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
        return common_bias_broadcast.gen_profiler(
            func_attrs,
            workdir,
            profiler_filename,
            dim_info_dict,
            RCR,
            unary_op1,
            binary_op1,
            binary_op2,
            unary_op2,
        )

    return gen_profiler


def gen_function_template(unary_op1, binary_op1, binary_op2, unary_op2):
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
            unary_op1,
            binary_op1,
            binary_op2,
            unary_op2,
        )

    return gen_function


def gen_function_decl(func_attrs):
    return common_bias_broadcast.gen_function_decl(func_attrs)


def gen_function_call(func_attrs, indent="  "):
    return common_bias_broadcast.gen_function_call(func_attrs, indent)


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


for conf in _CONFIGS:
    name, unary_op1, binary_op1, binary_op2, unary_op2 = conf
    registry.reg(f"cuda.gemm_rcr_bias_{name}.config")(gemm_rcr_config)
    registry.reg(f"cuda.gemm_rcr_bias_{name}.gen_profiler")(
        gen_profiler_template(unary_op1, binary_op1, binary_op2, unary_op2)
    )
    registry.reg(f"cuda.gemm_rcr_bias_{name}.gen_function")(
        gen_function_template(unary_op1, binary_op1, binary_op2, unary_op2)
    )
    registry.reg(f"cuda.gemm_rcr_bias_{name}.func_decl")(gen_function_decl)
    registry.reg(f"cuda.gemm_rcr_bias_{name}.func_call")(gen_function_call)
    registry.reg(f"cuda.gemm_rcr_bias_{name}.filter")(function_filter)
