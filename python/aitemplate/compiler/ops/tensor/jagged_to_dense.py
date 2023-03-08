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
Define jagged_to_dense op
"""
import logging
from typing import List

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntVar, Operator, Tensor

_LOGGER = logging.getLogger(__name__)


class jagged_to_dense(Operator):
    """
    Returns a tensor containing the dense format of the input jagged tensor.
    Args:
        x (Tensor): input jagged tensor
        padding_value (float): the padding value for elements out of jagged shape.
    Returns:
        y: a tensor containing the dense format of input jagged tensor.
    """

    def __init__(
        self,
        padding_value: float = 0,
    ):
        super().__init__()
        self._attrs["op"] = "jagged_to_dense"
        self._attrs["padding_value"] = padding_value

    def _infer_shape(self, x: Tensor) -> List[IntVar]:
        jagged_int_var = x.shape()[0]
        inner_shape = x.shape()[1:]
        return jagged_int_var.get_max_dense_shape() + inner_shape

    def _get_op_attributes(self):
        return {
            "padding_value": self._attrs["padding_value"],
        }

    def _args_for_pseudo_code(self):
        return [f"padding_value={self._attrs['padding_value']}"]

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        if not x.is_jagged():
            raise RuntimeError(
                "Input tensor x is expected to be jagged, but actually dense for jagged_to_dense."
            )

        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shape(x)
        y = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])

        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
