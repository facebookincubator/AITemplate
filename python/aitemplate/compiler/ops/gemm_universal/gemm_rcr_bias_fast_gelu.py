# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
gemm rcr with bias
"""
from . import gemm_rcr_bias

# pylint: disable=C0103,W0223,W0221


class gemm_rcr_bias_fast_gelu(gemm_rcr_bias):
    """_summary_

    Parameters
    ----------
    gemm_rcr : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_fast_gelu"
        self._attrs["epilogue"] = "LinearCombinationFastGELU"
