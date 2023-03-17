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
permute(0, 2, 1) op
"""
from typing import List

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0221


class permute021(Operator):
    """
    Permutes the input tensor from (B1, B2, ..., Bn, N, M) to (B1, B2, ..., Bn, M, N).

    Args:
        input (Tensor[B1, B2, ..., Bn, N, M]): the source tensor with 3 dimensions

    Returns:
        output (Tensor[B1, B2, ..., Bn, M, N]): the destination tensor

    Example:

        .. highlight:: python
        .. code-block:: python

            X = Tensor(shape=[2, 384, 262], name="X", is_input=True)
            Y = ops.permute021()(X)
            y_shape = [d._attrs["values"][0] for d in Y.shape()]
            print(y_shape)

            Outs:
            [2, 262, 384]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute021"

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for permute021."""
        x_shape = x._attrs["shape"]
        return x_shape[:-2] + [x_shape[-1], x_shape[-2]]

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape()) > 2, "The input tensor must have at least 3 dimensions"

        self._attrs["inputs"] = [x]
        self._attrs["input_accessors"] = [TensorAccessor(x)]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        output._attrs["dtype"] = x.dtype()
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
        )
