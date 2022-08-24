# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] fused transposed_conv2d_bias_relu op
"""
from .transposed_conv2d_bias import transposed_conv2d_bias

# pylint: disable=C0103
class transposed_conv2d_bias_relu(transposed_conv2d_bias):
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
        super().__init__(stride, pad, dilate=dilate, group=group)
        self._attrs["op"] = "transposed_conv2d_bias_relu"
        self._attrs["epilogue"] = "LinearCombinationRelu"
