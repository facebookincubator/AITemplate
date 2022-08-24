# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] bilinear_upsampling2d op
"""
from .upsampling2d import upsampling2d_base


# pylint: disable=C0103
class bilinear_upsampling2d(upsampling2d_base):
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
        self._attrs["op"] = "bilinear_upsampling2d"
