# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] avg_pool2d op
"""
from .pool2d import pool2d_base


# pylint: disable=C0103
class avg_pool2d(pool2d_base):
    """[summary]

    Parameters
    ----------
    pool2d_base : [type]
        [description]
    """

    def __init__(self, kernel_size, stride, pad) -> None:
        """[summary]

        Parameters
        ----------
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        pad : [type]
            [description]
        """
        super().__init__(stride, pad, kernel_size, "avg")
        self._attrs["op"] = "avg_pool2d"
