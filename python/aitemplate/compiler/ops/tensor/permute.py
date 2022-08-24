# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
permute op
"""
from typing import Sequence

from ....utils.tensor_utils import wrap_dim

from ...base import Operator, Tensor
from .permute021 import permute021
from .permute102 import permute102
from .permute210 import permute210


class permute(Operator):
    """
    Returns a tensor with its dimensions permuted. This returned tensor is not a view. Dim in dims can be negative.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute"

    def __call__(self, x: Tensor, dims: Sequence[int]) -> Tensor:
        if len(dims) != 3:
            raise NotImplementedError(
                "Permute op doesn't support permute pattern {}".format(dims)
            )
        dims = list(dims)
        for i, dim in enumerate(dims):
            dims[i] = wrap_dim(dim, 3)

        if dims == [0, 2, 1]:
            return permute021()(x)
        if dims == [1, 0, 2]:
            return permute102()(x)
        if dims == [2, 1, 0]:
            return permute210()(x)
        raise NotImplementedError(
            "Permute op doesn't support permute pattern {}".format(dims)
        )
