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

from aitemplate.compiler import compile_model, ops

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


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


class OpCommonElementwiseTestCase(unittest.TestCase):
    def test_elementwise_type_promotion_bool_rhs(self):
        X0 = Tensor(
            shape=[3, 5, 2],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[3, 5, 2],
            dtype="bool",
            name="X1",
            is_input=True,
        )
        Y = ops.elementwise(FuncEnum.MUL)(X0, X1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            "test_elementwise_type_promotion_bool_rhs",
        )
        x0_pt = get_random_torch_tensor([3, 5, 2], "float16")
        x1_pt = get_random_torch_tensor([3, 5, 2], "bool")
        out_pt = get_torch_empty_tensor([3, 5, 2], "float16")
        module.run_with_tensors({"X0": x0_pt, "X1": x1_pt}, {"output0": out_pt})

    def test_elementwise_type_promotion_bool_lhs(self):
        X0 = Tensor(
            shape=[3, 5, 2],
            dtype="bool",
            name="X1",
            is_input=True,
        )
        X1 = Tensor(
            shape=[3, 5, 2],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        Y = ops.elementwise(FuncEnum.MUL)(X0, X1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            "test_elementwise_type_promotion_bool_lhs",
        )
        x0_pt = get_random_torch_tensor([3, 5, 2], "float16")
        x1_pt = get_random_torch_tensor([3, 5, 2], "bool")
        out_pt = get_torch_empty_tensor([3, 5, 2], "float16")
        module.run_with_tensors({"X0": x0_pt, "X1": x1_pt}, {"output0": out_pt})
