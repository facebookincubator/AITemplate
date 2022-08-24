# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
