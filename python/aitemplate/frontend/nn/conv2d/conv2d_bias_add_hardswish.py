# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] conv2d + bias + residual + hardswish
"""
from .common_conv2d_bias_add_act import Conv2dBiasAddAct


class Conv2dBiasAddHardswish(Conv2dBiasAddAct):
    """[summary]

    Parameters
    ----------
    Block : [type]
        [description]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
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
            "conv2d_bias_add_hardswish",
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
        )
