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
from aitemplate.compiler.ops.common import elementwise
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.gemm_universal import (
    bmm_ccr,
    bmm_ccr_add,
    bmm_crr,
    bmm_crr_add,
    bmm_rrr,
    bmm_rrr_add,
    gemm_rcr,
    gemm_rcr_bias,
    gemm_rcr_bias_add,
    gemm_rcr_bias_add_add,
    gemm_rcr_bias_add_add_relu,
    gemm_rcr_bias_add_relu,
    gemm_rcr_bias_fast_gelu,
    gemm_rcr_bias_gelu,
    gemm_rcr_bias_mul,
    gemm_rcr_bias_mul_add,
    gemm_rcr_bias_mul_tanh,
    gemm_rcr_bias_relu,
    gemm_rcr_bias_sigmoid,
    gemm_rcr_bias_sigmoid_mul,
    gemm_rcr_bias_sigmoid_mul_tanh,
    gemm_rcr_bias_tanh,
)


def get_gemm_rcr_bias_patterns():
    gemm_rcr_bias_patterns = [
        (
            (gemm_rcr(), elementwise(FuncEnum.ADD)),
            gemm_rcr_bias,
        ),
    ]
    return gemm_rcr_bias_patterns


def get_patterns():
    """
    We create the pattern of fusion here.
    The format should be in the form of (pattern, replacement)

    pattern: This would be a list of operator which are chained which we
             want to match
    replacement: The op to replace pattern.
    """
    bmm_ccr_patterns = [
        ((bmm_ccr(), elementwise(FuncEnum.ADD)), bmm_ccr_add),
    ]

    bmm_crr_patterns = [
        ((bmm_crr(), elementwise(FuncEnum.ADD)), bmm_crr_add),
    ]

    bmm_rrr_patterns = [
        ((bmm_rrr(), elementwise(FuncEnum.ADD)), bmm_rrr_add),
    ]

    gemm_rcr_bias_activation_patterns = [
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.RELU),
            ),
            gemm_rcr_bias_relu,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.SIGMOID),
                elementwise(FuncEnum.MUL),
                elementwise(FuncEnum.TANH),
            ),
            gemm_rcr_bias_sigmoid_mul_tanh,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.SIGMOID),
                elementwise(FuncEnum.MUL),
            ),
            gemm_rcr_bias_sigmoid_mul,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.SIGMOID),
            ),
            gemm_rcr_bias_sigmoid,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.TANH),
            ),
            gemm_rcr_bias_tanh,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.GELU),
            ),
            gemm_rcr_bias_gelu,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.FASTGELU),
            ),
            gemm_rcr_bias_fast_gelu,
        ),
    ]

    gemm_rcr_bias_add_patterns = [
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            gemm_rcr_bias_add_relu,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            gemm_rcr_bias_add_add_relu,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.ADD),
            ),
            gemm_rcr_bias_add_add,
        ),
        (
            (gemm_rcr_bias(), elementwise(FuncEnum.ADD)),
            gemm_rcr_bias_add,
        ),
    ]

    gemm_rcr_bias_mul_patterns = [
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.MUL),
                elementwise(FuncEnum.ADD),
            ),
            gemm_rcr_bias_mul_add,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.MUL),
                elementwise(FuncEnum.TANH),
            ),
            gemm_rcr_bias_mul_tanh,
        ),
        (
            (
                gemm_rcr_bias(),
                elementwise(FuncEnum.MUL),
            ),
            gemm_rcr_bias_mul,
        ),
    ]

    fusion_patterns = (
        bmm_ccr_patterns
        + bmm_crr_patterns
        + bmm_rrr_patterns
        + gemm_rcr_bias_activation_patterns
        + gemm_rcr_bias_add_patterns
        + gemm_rcr_bias_mul_patterns
    )

    return fusion_patterns
