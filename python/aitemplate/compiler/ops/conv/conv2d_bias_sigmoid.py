# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] fused conv2d_bias_sigmoid op
"""
from .common_conv2d_bias_activation import conv2d_bias_activation


# pylint: disable=C0103
class conv2d_bias_sigmoid(conv2d_bias_activation):
    """[summary]

    Parameters
    ----------
    conv2d : [type]
        [description]
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """[summary]

        Parameters
        ----------
        stride : [type]
            [description]
        pad : [type]
            [description]
        dilate : int, optional
            [description], by default 1
        """
        super().__init__("sigmoid", stride, pad, dilate=dilate, group=group)
        self._attrs["epilogue"] = "LinearCombinationSigmoid"
