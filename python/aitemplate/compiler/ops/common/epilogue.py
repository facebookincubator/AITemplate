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
Epilogue definitions.
TODO(ipiszy): Rename it to types.py.
"""

from enum import Enum


class EpilogueOp(Enum):
    """
    Epilogue enum.
    """

    NA = 1
    BIAS = 2
    BIAS_RELU = 3
    BIAS_RELU_ADD = 4


class FuncEnum(Enum):
    """
    Elementwise func enum.
    """

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    TANH = 5
    COS = 6
    SIN = 7
    SIGN = 8
    ABS = 9
    LOGE = 10
    EXP = 11
    SQRT = 12
    MAX = 13
    MIN = 14
    SIGMOID = 15
    LRELU = 16
    HARDTANH = 17
    RELU = 18
    NAN_TO_NUM = 19
    CLAMP_NAN_TO_NUM = 20
    SILU = 21
    POW = 22
    GELU = 23
    FASTGELU = 24
    SOFTPLUS = 25
    ELU = 26
    SOFTSIGN = 27
    FLOOR_DIV = 28
    CELU = 29
