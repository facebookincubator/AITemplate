# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.ops import avg_pool2d, max_pool2d
from .module import Module


class MaxPool2d(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.op = max_pool2d(kernel_size, stride, padding)

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


class AvgPool2d(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self, kernel_size, stride, padding):
        """[summary]

        Parameters
        ----------
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        padding : [type]
            [description]
        """
        super().__init__()
        self.op = avg_pool2d(kernel_size, stride, padding)

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
