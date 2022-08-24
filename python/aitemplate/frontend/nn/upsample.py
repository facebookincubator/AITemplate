# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.ops import bilinear_upsampling2d, bilinear_upsampling2d_add
from .module import Module


class BilinearUpsampling2d(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self, scale_factor, mode):
        """[summary]

        Parameters
        ----------
        scale_factor : [type]
            [description]
        """
        super().__init__()
        self.op = bilinear_upsampling2d(scale_factor, mode)

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


class BilinearUpsampling2dAdd(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self, scale_factor, mode):
        """[summary]

        Parameters
        ----------
        scale_factor : [type]
            [description]
        """
        super().__init__()
        self.op = bilinear_upsampling2d_add(scale_factor, mode)

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 2
        x = args[0]
        res = args[1]
        return self.op(x, res)
