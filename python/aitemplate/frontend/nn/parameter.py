# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.base import Tensor


class Parameter(object):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self, shape, dtype, name=None, value=None):
        """[summary]

        Parameters
        ----------
        shape : [type]
            [description]
        dtype : [type]
            [description]
        name : [type], optional
            [description], by default None
        value : [type], optional
            [description], by default None
        """
        self._tensor = Tensor(shape=shape, dtype=dtype, name=name)
        self._value = value

    def tensor(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._tensor

    def value(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._value
