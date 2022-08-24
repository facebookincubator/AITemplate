# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] bilinear_upsampling2d_add op
"""
from typing import List

from ...base import Tensor
from .upsampling2d import upsampling2d_base


# pylint: disable=C0103
class bilinear_upsampling2d_add(upsampling2d_base):
    """[summary]

    Parameters
    ----------
    upsampling2d_base : [type]
        [description]
    """

    def __init__(self, scale_factor, mode) -> None:
        """[summary]

        Parameters
        ----------
        scale_factor : [type]
            [description]
        """
        super().__init__(scale_factor, mode)
        self._attrs["op"] = "bilinear_upsampling2d_add"

    def __call__(self, x: Tensor, r: Tensor) -> List[Tensor]:
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]

        Returns
        -------
        List[Tensor]
            [description]
        """
        self._attrs["inputs"] = [x, r]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output
