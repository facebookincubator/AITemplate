# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] fused conv2d_bias_relu op
"""
from .special_conv2d_bias_activation import special_conv2d_bias_activation

# pylint: disable=C0103
class conv2d_bias_relu_few_channels(special_conv2d_bias_activation):
    """conv2d_bias_relu_few_channels operator"""

    def __init__(self, stride, pad, dilate=1, auto_padding=True) -> None:
        """Initializes conv2d_bias_relu_few_channels"""
        super().__init__("relu", stride, pad, dilate, auto_padding)
