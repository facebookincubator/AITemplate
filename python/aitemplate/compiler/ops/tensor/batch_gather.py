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
Batch_gather.
"""
import itertools
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
M == {{x_dim0}} && K == {{x_dim1}}
"""
)


class batch_gather(Operator):
    """
    Gathers values of the `input` tensor specified by `indicies`. Dim 0 of `indicies` correspond to the indices of `input` elements in dim 0.

    Args:
        input (Tensor): the source tensor
        indices (Tensor): the indices of elements to gather

    Returns:
        Tensor: the destination tensor
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "batch_gather"
        self._attrs["has_profiler"] = False
        self.exec_key_template = EXEC_KEY_TEMPLATE

    def _infer_shape(self, x: List[int], indices: List[int]):
        rank = len(indices)
        for r in range(rank - 1):
            assert x[r] == indices[r]
        output = list(x)
        output[rank - 1] = indices[-1]
        return output

    def _infer_shapes(self, x: Tensor, indices: Tensor) -> List[IntVar]:
        """Infers shapes for batch_gather."""

        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        indices_shape = [var._attrs["values"][0] for var in indices._attrs["shape"]]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, indices_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = []
        for idx in range(len(y_shapes[0])):
            output_shape.append(
                shape_utils.gen_int_var(unique([d[idx] for d in y_shapes]))
            )
        return output_shape

    def __call__(self, x: Tensor, indices: Tensor) -> Tensor:
        dtype = indices._attrs["dtype"]
        assert dtype in [
            "int",
            "int32",
            "int64",
        ], f"batch_gather(): Expected dtype int/int32/int64 for index, got dtype {dtype}"
        self._attrs["inputs"] = [x, indices]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x, indices)
        output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())
        self._attrs["outputs"] = [output]
        return output

    def _gen_exec_key(self, shape):
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
