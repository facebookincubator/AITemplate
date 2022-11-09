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
permute op
"""
from typing import List, Sequence

from .... import backend
from ....backend import registry
from ....utils.tensor_utils import wrap_dim
from ...base import IntVar, Operator, Tensor
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

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for permute."""

        output_shapes = []
        input_shapes = x.shape()
        for dim in self._attrs["dims"]:
            output_shapes.append(input_shapes[dim])
        return output_shapes

    def __call__(self, x: Tensor, dims: Sequence[int]) -> Tensor:
        dims = list(dims)
        for i, dim in enumerate(dims):
            dims[i] = wrap_dim(dim, x._rank())

        if dims == [0, 2, 1]:
            return permute021()(x)
        if dims == [1, 0, 2]:
            return permute102()(x)
        if dims == [2, 1, 0]:
            return permute210()(x)

        self._attrs["dims"] = dims
        self._attrs["inputs"] = [x]
        self._set_depth()

        output_shapes = self._infer_shapes(x)
        output = Tensor(output_shapes, src_ops={self})
        self._attrs["outputs"] = [output]

        # TODO: support output TensorAccessor
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
        )
