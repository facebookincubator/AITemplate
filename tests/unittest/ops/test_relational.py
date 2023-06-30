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

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.public import FuncEnum
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    gen_input_tensor,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from parameterized import param, parameterized

ait_to_torch_map = {
    ops.ge: torch.ge,
    ops.le: torch.le,
    ops.gt: torch.gt,
    ops.lt: torch.lt,
    ops.eq: torch.eq,
    ops.ne: torch.ne,
}


def get_test_cases(dtype: str):
    return [
        param(ops.le, "le", dtype, 3),
        param(ops.lt, "lt", dtype, 4),
    ]


class TestRelational(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    param(ops.le, "le", "float16", 3),
                    param(ops.le, "lt", "float16", 3),
                ],
                TestEnv.CUDA_SM80: [
                    param(ops.le, "le", "bfloat16", 3),
                    param(ops.le, "lt", "bfloat16", 3),
                    param(ops.gt, "gt", "float32", 8),
                    param(ops.ne, "ne", "float", 1),
                    param(ops.eq, "eq", "float", 16),
                ],
                TestEnv.ROCM: [
                    param(ops.le, "le", "float16", 3),
                    param(ops.le, "lt", "float16", 3),
                ],
            }
        )
    )
    def test_end_to_end(
        self, operator: type, test_name: str, dtype: str, M: int
    ) -> None:
        dim = IntVar([2, 3, 128])
        X1 = gen_input_tensor([dim, M], name="X1", dtype=dtype)
        X2 = gen_input_tensor([dim, M], name="X2", dtype=dtype)
        add = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X3 = gen_input_tensor([dim, M], name="X3", dtype=dtype)
        Y = operator()(add, X3)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", f"test_relational_{test_name}")

        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            x2_pt = get_random_torch_tensor([batch, M], dtype)
            x3_pt = get_random_torch_tensor([batch, M], dtype)
            add_pt = x1_pt + x2_pt
            y_pt = ait_to_torch_map[operator](add_pt, x3_pt)
            y = get_torch_empty_tensor(y_pt.size(), dtype="bool")
            inputs = {"X1": x1_pt, "X2": x2_pt, "X3": x3_pt}
            model.run_with_tensors(inputs, [y])
            self.assertEqual(y_pt.tolist(), y.tolist())

    def test_unsupport_type_promotion(self) -> None:
        dim = IntVar([1, 128])
        X1 = Tensor([dim, 10], name="X1", is_input=True, dtype="float16")
        X2 = Tensor([dim, 10], name="X2", is_input=True, dtype="float32")
        with self.assertRaisesRegex(
            AssertionError, "Type promotions are not supported"
        ):
            ops.ge()(X1, X2)

    def test_unsupport_different_shapes(self) -> None:
        X1 = Tensor([IntVar([1, 128]), 10], name="X1", is_input=True, dtype="float16")
        X2 = Tensor([IntVar([10, 128]), 10], name="X2", is_input=True, dtype="float16")
        with self.assertRaisesRegex(
            AssertionError,
            "Relational does not support broadcasting yet. It expects tensor of same shape",
        ):
            ops.ge()(X1, X2)

    def test_constant(self) -> None:
        X1 = Tensor([IntVar([1, 128]), 10], name="X1", is_input=True, dtype="float16")
        X2 = 2
        Y = ops.ge()(X1, X2)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", "test_relational_test_constant")

        x1_pt = get_random_torch_tensor([128, 10], dtype="float16")
        inputs = {"X1": x1_pt}
        y_pt = ait_to_torch_map[ops.ge](x1_pt, 2)
        y = get_torch_empty_tensor(y_pt.size(), dtype="bool")
        model.run_with_tensors(inputs, [y])
        self.assertEqual(y_pt.tolist(), y.tolist())

    @parameterized.expand(
        [
            param("int32", 3),
            param("int32", 2),
            param("int64", 3),
            param("int64", 2),
        ]
    )
    def test_int_support(self, dtype: str, M: int) -> None:
        dim = IntVar([2, 3, 128])
        X1 = gen_input_tensor([dim, M], name="X1", dtype=dtype)
        X2 = gen_input_tensor([dim, M], name="X2", dtype=dtype)
        Y = ops.ge()(X1, X2)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", f"test_relational_int_{dtype}_{M}")

        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], "float32").to(
                torch.int32 if dtype == "int32" else torch.int64
            )
            x2_pt = get_random_torch_tensor([batch, M], "float32").to(
                torch.int32 if dtype == "int32" else torch.int64
            )
            y_pt = ait_to_torch_map[ops.ge](x1_pt, x2_pt)
            y = get_torch_empty_tensor(y_pt.size(), dtype="bool")
            inputs = {"X1": x1_pt, "X2": x2_pt}
            model.run_with_tensors(inputs, [y])
            self.assertEqual(y_pt.tolist(), y.tolist())


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
