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

    def _infer_shapes(self, x: Tensor, indices: Tensor) -> List[IntVar]:
        """Infers shapes for batch_gather."""

        rank = len(indices._attrs["shape"])

        # TODO: remove this when we're sure we support non-static batch_gather
        x_shape_values = [var._attrs["values"][0] for var in x._attrs["shape"]]
        indices_shape = [var._attrs["values"][0] for var in indices._attrs["shape"]]
        for r in range(1, rank - 1):
            assert x_shape_values[r] == indices_shape[r]

        out_shapes = x._attrs["shape"][:]
        if rank <= 1:
            # Special case: gather happens along batch dimension
            out_shapes[0] = indices.shape()[0]
        out_shapes[rank - 1] = indices._attrs["shape"][-1]

        return out_shapes

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
