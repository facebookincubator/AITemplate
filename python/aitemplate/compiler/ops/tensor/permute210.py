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
Permute(2, 1, 0) op.
Swap the dimension of dim0 and dim2 of input 3d tensor.
"""
from typing import List

from aitemplate import backend

from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221


class permute210(Operator):
    """
    Permutes the input 3d tensor from (B, N, M) to (M, N, B).

    Args:
        input (Tensor[B, N, M]): the source tensor with 3 dimensions

    Returns:
        output (Tensor[M, N, B]): the destination tensor

    Example:

        .. highlight:: python
        .. code-block:: python

            X = Tensor(shape=[2, 384, 262], name="X", is_input=True)
            Y = ops.permute210()(X)
            y_shape = [d._attrs["values"][0] for d in Y.shape()]
            print(y_shape)

            Outs:
            [262, 384, 2]

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute210"

    def _infer_shapes(self, x: Tensor):
        """Infers shapes for permute210.

        Parameters
        ----------
        x : Tensor

        Returns
        ------
        Tensor
            Inferred output 3d tensor with input shape.
            Because its a permute210 operation,
            Out.d0=In.d2, Out.d2=In.d0.
        """
        x_shape = x._attrs["shape"]
        return [x_shape[2], x_shape[1], x_shape[0]]

    def __call__(self, x: Tensor) -> List[Tensor]:
        """
        Return the output tensor of permute210

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            Generate output tensors of function calls.
            In permute210, its a 3d tensor with d2,d1,d0 of
            input Tensor.
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        output._attrs["dtype"] = x.dtype()
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """
        Generate function body

        Returns
        -------
        str
           The function body string
        """
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
