# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] fused conv2d_bias_relu_add op, for residual block
"""
from ...base import Tensor
from .conv2d import conv2d


# pylint: disable=C0103
class conv2d_bias_add_activation(conv2d):
    """[summary]

    Parameters
    ----------
    conv2d : [type]
        [description]
    """

    def __init__(self, activation, stride, pad, dilate=1, group=1) -> None:
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
        self._attrs["op"] = "conv2d_bias_add_{act}".format(act=activation)
        self._attrs["epilogue"] = "LinearCombinationResidualBlock"

    def __call__(self, x: Tensor, w: Tensor, b: Tensor, r: Tensor):
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]
        b : Tensor
            [description]
        r : Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._attrs["inputs"] = [x, w, b, r]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self})
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output
