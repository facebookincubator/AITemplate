# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
var op implementation
"""
from .reduce_common import reduce_base

# pylint: disable=C0103


class var(reduce_base):
    """a class that implements the var op

    Parameters
    ----------
    reduce_base : reduce_base class
        the base class that var class derives from
    """

    def __init__(self, dim, unbiased, keepdim=False, dtype=None) -> None:
        """initialization routine of var op

        Parameters
        ----------
        dim : int or tuple of python:ints
            the dimension or dimensions to reduce
        unbiased : bool
            specifying whether to use Bessel’s correction or not
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
        self._attrs["op"] = "var"
        self._attrs["unbiased"] = unbiased
