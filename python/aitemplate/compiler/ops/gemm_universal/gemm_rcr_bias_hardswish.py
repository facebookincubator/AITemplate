# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
gemm rcr with bias
"""
from . import gemm_rcr_bias

# pylint: disable=C0103,W0223,W0221


class gemm_rcr_bias_hardswish(gemm_rcr_bias):
    """
    gemm_rcr_bias_hardswish

    x = gemm_rcr_bias(inputs)
    out = x * relu6(x+3) / 6
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_hardswish"
        self._attrs["epilogue"] = "LinearCombinationHardSwish"
