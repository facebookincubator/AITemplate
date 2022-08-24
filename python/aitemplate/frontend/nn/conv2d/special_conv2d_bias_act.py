# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] common module for conv_bias_act subgraph
"""
from ....compiler import ops
from ..module import Module
from ..parameter import Parameter

# pylint: disable=C0103


class SpecialConv2dBiasAct(Module):
    """common functions for conv2d_bias_act op with few channel input"""

    def __init__(
        self,
        op_name,
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
        super().__init__()
        if auto_padding and in_channels < 4:
            in_channels = 4
        elif auto_padding and in_channels > 4 and in_channels < 8:
            in_channels = 8
        self.W = Parameter(
            shape=[out_channels, kernel_size, kernel_size, in_channels], dtype=dtype
        )
        self.B = Parameter(shape=[out_channels], dtype=dtype)
        op_func = getattr(ops, op_name)
        self.op = op_func(
            stride=stride, pad=padding, dilate=dilation, auto_padding=auto_padding
        )

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 1
        x = args[0]
        return self.op(x, self.W.tensor(), self.B.tensor())
