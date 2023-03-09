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

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.ops.tensor.permute021 import permute021
from aitemplate.compiler.ops.tensor.permute0213 import permute0213
from aitemplate.compiler.ops.tensor.permute102 import permute102
from aitemplate.compiler.ops.tensor.permute210 import permute210
from aitemplate.utils.tensor_utils import wrap_dim


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

        sorted_dims = list(range(x._rank()))
        assert (
            sorted(dims) == sorted_dims
        ), f"expected a permutation of {sorted_dims}, but got {dims}"

        # "dims" is set here before possible dispatching to the
        # static-shape permute kernels below to keep the call to
        # ops.permute(..., dims) recoverable from the self._attrs
        self._attrs["dims"] = dims

        if dims == [0, 2, 1]:
            return permute021()(x)
        if dims == [1, 0, 2]:
            return permute102()(x)
        if dims == [2, 1, 0]:
            return permute210()(x)

        if dims == [0, 2, 1, 3]:
            second_dim = x.shape()[1]
            if (isinstance(second_dim, IntImm) and second_dim.value() >= 24) or (
                isinstance(second_dim, IntVar) and second_dim.lower_bound() >= 24
            ):
                # for (0, 2, 1, 3) dims, we dispatch to the permute0213 op
                # when the second dim >= 24 due to a better performance
                return permute0213()(x)

        last_dim = x.shape()[-1]
        if (
            len(dims) > 3
            and dims[:-2] + [dims[-1], dims[-2]] == sorted_dims
            and (
                (isinstance(last_dim, IntImm) and last_dim.value() >= 8)
                or (isinstance(last_dim, IntVar) and last_dim.lower_bound() >= 8)
            )
        ):
            # when swapping the last two dims and the last_dim >= 8, we
            # dispatch to the permute021 op due to a better performance
            return permute021()(x)

        self._attrs["inputs"] = [x]
        self._set_depth()

        output_shapes = self._infer_shapes(x)
        output = Tensor(output_shapes, src_ops={self})
        self._attrs["outputs"] = [output]
        output._attrs["dtype"] = x.dtype()

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
