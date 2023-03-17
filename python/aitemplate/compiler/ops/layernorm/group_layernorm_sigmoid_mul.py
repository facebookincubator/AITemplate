#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Operator definition for group_layernorm_sigmoid_mul.
"""
from typing import List

from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.layernorm.group_layernorm import group_layernorm

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
        super().__init__(normalized_shape)
        self._attrs["op"] = "group_layernorm_sigmoid_mul"
        self._attrs["has_profiler"] = False
