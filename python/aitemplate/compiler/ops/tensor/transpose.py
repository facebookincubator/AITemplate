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
transpose op
"""

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.tensor.permute import permute


class transpose(permute):
    """
    Returns a tensor with its two dimensions transposed.
    This returned tensor is not a view. Dims can be negative.
    """

    def __call__(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        dims = list(range(x._rank()))
        dims[dim0] = dim1
        dims[dim1] = dim0

        return super().__call__(x, dims)
