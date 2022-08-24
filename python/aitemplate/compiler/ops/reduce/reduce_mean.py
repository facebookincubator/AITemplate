# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] reduce_mean op implementation
"""
from .reduce_common import reduce_base

# pylint: disable=C0103


class reduce_mean(reduce_base):
    """a class that implements the reduce_mean op

    Parameters
    ----------
    reduce_base : reduce_base class
        the base class that reduce_mean derives from
    """

    def __init__(self, dim, keepdim=False, dtype=None) -> None:
        """[summary]

        Parameters
        ----------
        dim : int or tuple of python:ints
            the dimension or dimensions to reduce
        keepdim : bool, optional
            keep the reduced dimensions if True, default is False
        dtype : str, optional
            the type of the return tensor. If it is not None,
            the input tensor is cast to dtype before reduction.

        Return
        ----------
        None

        """
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "reduce_mean"
