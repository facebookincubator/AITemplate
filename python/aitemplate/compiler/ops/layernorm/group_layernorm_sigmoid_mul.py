# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from typing import List

from ...base import IntImm
from .group_layernorm import group_layernorm

# pylint: disable=C0103,W0221,W0102,W0223


class group_layernorm_sigmoid_mul(group_layernorm):
    """group_layernorm_sigmoid_mul.
    For each group, we expect each input to have shapes:
        Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
        Normalized_shape: [N1, N2, ..., ND]
        Gamma/Beta, if not None, have the same shape as normalized_shape.
    Every input in the groups must have the same [M0, M1, ..., Mp] dims.
    """

    def __init__(self, normalized_shape: List[List[IntImm]] = None) -> None:
        """[summary]

        Parameters
        ----------
        """
        super().__init__(normalized_shape)
        self._attrs["op"] = "group_layernorm_sigmoid_mul"
        self._attrs["has_profiler"] = False
