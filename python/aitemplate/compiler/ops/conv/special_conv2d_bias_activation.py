# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] fused conv2d_bias_relu op
"""
from ...base import Tensor
from ..padding import nhwc3to8, pad_last_dim
from .conv2d import conv2d

# pylint: disable=C0103
class special_conv2d_bias_activation(conv2d):
    """[summary]

    Parameters
    ----------
    conv2d : [type]
        [description]
    """

    def __init__(self, activation, stride, pad, dilate=1, auto_padding=True) -> None:
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
        super().__init__(stride, pad, dilate=dilate)
        self._attrs["op"] = "conv2d_bias_{act}_few_channels".format(act=activation)
        self._attrs["epilogue"] = "LinearCombinationRelu"
        self._auto_padding = auto_padding

    def __call__(self, x: Tensor, w: Tensor, b: Tensor):
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]
        b : Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        if self._auto_padding:
            last_dim = x._attrs["shape"][-1]._attrs["values"][0]
            if last_dim in range(1, 4):
                x = pad_last_dim(len(x._attrs["shape"]), 4)(x)
            elif last_dim in range(5, 8):
                x = nhwc3to8()(x)
        self._attrs["inputs"] = [x, w, b]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self})
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output
