# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] reduce_sum op
"""
from .reduce_common import reduce_base

# pylint: disable=C0103


class reduce_sum(reduce_base):
    """[summary]

    Parameters
    ----------
    reduce_base : [type]
        [description]
    """

    def __init__(self, dim, keepdim=False, dtype=None) -> None:
        """[summary]

        Parameters
        ----------
        dim : int or tuple of python:ints
            [description] the dimension or dimensions to reduce
        keepdim : bool
            [description] keep the reduced dimensions if True, default is False
        dtype : optional str
            [description] the type of the return tensor. If it is not None,
                          the input tensor is casted to dtype before reduction.
        """
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "reduce_sum"
