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
identity op
"""
from typing import List

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor


class identity(Operator):
    """
    Returns the input tensor. This could be useful for only name changes etc.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "identity"

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        return x.shape()

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()

        output_shapes = self._infer_shapes(x)
        output = Tensor(output_shapes, src_ops={self}, is_view_of=x)
        self._attrs["outputs"] = [output]

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
