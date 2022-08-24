# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
tanh(sigmoid((gemm(A, B) + bias)) * D0)
"""

from .gemm_rcr_bias_broadcast import gemm_rcr_bias_broadcast

# pylint: disable=C0103, W0223, W0221


class gemm_rcr_bias_sigmoid_mul_tanh(gemm_rcr_bias_broadcast):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_sigmoid_mul_tanh"
        self._attrs["num_sources"] = 1
