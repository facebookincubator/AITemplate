# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.ops import flatten, reshape
from .module import Module


class Reshape(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class View(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class Flatten(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
        [description]
    """

    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.op = flatten(start_dim, end_dim)

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
