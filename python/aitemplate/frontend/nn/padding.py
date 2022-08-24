# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.ops import nhwc3to8
from .module import Module


class Nhwc3to8(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self):
        super().__init__()
        self.op = nhwc3to8()

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 1
        x = args[0]
        return self.op(x)
