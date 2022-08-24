# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] conv2d bias relu module
"""
from .special_conv2d_bias_act import SpecialConv2dBiasAct


class Conv2dBiasReluFewChannels(SpecialConv2dBiasAct):
    """functions for the op with conv2d+bias+relu pattern"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        auto_padding=True,
        dtype="float16",
    ):
        """[summary]

        Parameters
        ----------
        in_channel : [type]
            [description]
        out_channel : [type]
            [description]
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        pad : str, optional
            [description], by default 'SAME'
        dilate : int, optional
            [description], by default 1
        dtype : str, optional
            [description], by default "float16"

        Raises
        ------
        NotImplementedError
            [description]
        """
        super().__init__(
            "conv2d_bias_relu_few_channels",
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            auto_padding,
            dtype,
        )
