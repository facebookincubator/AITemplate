# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
vector_norm op implementation that simulates pytorch's linalg.vector_norm.
Currently, we only support L2 norm.
"""
from .reduce_common import reduce_base

# pylint: disable=C0103


class vector_norm(reduce_base):
    """a class that implements the vector_norm op

    Parameters
    ----------
    reduce_base : reduce_base class
        the base class that vector_norm derives from
    """

    def __init__(self, ord_kind=2, dim=None, keepdim=False, dtype=None) -> None:
        """initialize the op

        Parameters
        ----------
        ord_kind : int or float or str, optional
            specifies the vector norm to be computed. (default: 2)
        dim : None or int or tuple of python:ints, optional
            the dimension or dimensions to be normalized.
            (default: None, in this case the input tensor will be treated as
             a 1-D tensor)
        keepdim : bool, optional
            keep the normalized dimensions if True, default is False
        dtype : str, optional
            the type of the return tensor. If it is not None,
            the input tensor is cast to dtype before reduction.

        Returns
        ----------
            None
        """
        if dim is None:
            raise NotImplementedError(
                "flattening input tensor before normalization is not supported yet"
            )
        super().__init__(dim, keepdim, dtype)
        self._attrs["op"] = "vector_norm"
        self._attrs["ord_kind"] = str(ord_kind)
