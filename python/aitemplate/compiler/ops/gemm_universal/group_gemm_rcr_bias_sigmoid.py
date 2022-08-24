# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""[summary]
"""

from . import group_gemm_rcr_bias

# pylint: disable=C0103,W0223


class group_gemm_rcr_bias_sigmoid(group_gemm_rcr_bias):
    """_summary_

    Parameters
    ----------
    group_gemm_rcr_bias : _type_
        _description_
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "group_gemm_rcr_bias_sigmoid"
        self._attrs["epilogue"] = "LinearCombinationSigmoid"
