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
View-related modules.
"""
from aitemplate.compiler.ops import flatten, reshape
from aitemplate.frontend.nn.module import Module


class Reshape(Module):
    """
    Returns a tensor with the same data and number of elements as input, but with the
    specified shape. Inputs must be contiguous.

    A single dimension may be -1, in which case it’s inferred from the remaining
    dimensions and the number of elements in input.
    """

    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        """Reshaped the input to given size."""
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class View(Module):
    """
    Placeholder for View layer. The current implementation is the same as Reshape.
    Returns a tensor with the same data and number of elements as input, but with the specified shape. Inputs must be contiguous.

    A single dimension may be -1, in which case it’s inferred from the remaining
    dimensions and the number of elements in input.
    """

    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        """Creates a view (copy) of the input with given shape."""
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class Flatten(Module):
    """
    Flattens input by reshaping it into a one-dimensional tensor. If start_dim or end_dim
    are passed, only dimensions starting with start_dim and ending with end_dim are
    flattened. The order of elements in input is unchanged.
    """

    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.op = flatten(start_dim, end_dim)

    def forward(self, *args):
        """Flattens the input with specified start and end dims."""
        assert len(args) == 1
        x = args[0]
        return self.op(x)
