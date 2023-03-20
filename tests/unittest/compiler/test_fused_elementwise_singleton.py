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
import unittest

from aitemplate import compiler

from aitemplate.compiler import ops

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.fuse_ops import (
    fuse_elementwise,
    process_singleton_elementwise,
)
from aitemplate.testing import detect_target


def _make_graph():
    X0 = Tensor(
        shape=[3, 5, 7, 9],
        dtype="float16",
        name="X0",
        is_input=True,
    )

    Y = ops.elementwise(FuncEnum.ABS)(ops.elementwise(FuncEnum.SIN)(X0))

    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    return Y


class FusedElementwiseSingletonTestCase(unittest.TestCase):
    def test_singleton_elementwise(self):
        Y = _make_graph()

        with detect_target():
            graph = compiler.transform.toposort(Y)
            compiler.transform.name_graph(graph)
            g1 = process_singleton_elementwise(graph)

        self.assertEqual(3, len(g1))  # x, sin(x), abs(sin(x))

    def test_fused_elementwise(self):
        Y = _make_graph()

        with detect_target():
            graph = compiler.transform.toposort(Y)
            compiler.transform.name_graph(graph)
            g1 = fuse_elementwise(graph)

        self.assertEqual(2, len(g1))  # x, abs(sin(x))
