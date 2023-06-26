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
from aitemplate.frontend import IntVar
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    gen_input_tensor,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from parameterized import param, parameterized


class TestWhere(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def test_unsupport_condition_tensor_non_bool(self) -> None:
        X1 = gen_input_tensor([4, 4], name="X1", dtype="float")
        X2 = gen_input_tensor([4, 4], name="X2", dtype="float")
        X3 = gen_input_tensor([4, 4], name="X3", dtype="float")
        with self.assertRaisesRegex(
            AssertionError, "condition needs to be a bool tensor"
        ):
            ops.where()(X1, X2, X3)

    def test_unsupport_condition_tensor_constant(self) -> None:
        X1 = 1
        X2 = gen_input_tensor([4, 4], name="X2", dtype="float")
        X3 = gen_input_tensor([4, 4], name="X3", dtype="float")
        with self.assertRaisesRegex(AssertionError, "condition needs to be a tensor"):
            ops.where()(X1, X2, X3)

    def test_unsupport_different_condition_and_input_tensor_size(self) -> None:
        dim = IntVar([2, 128])
        X1 = gen_input_tensor([dim, 4], name="X1", dtype="bool")
        X2 = gen_input_tensor([4, 4], name="X2", dtype="float")
        X3 = gen_input_tensor([4, 4], name="X3", dtype="float")
        with self.assertRaisesRegex(AssertionError, "Tensor shape should be the same"):
            ops.where()(X1, X2, X3)

    def test_unsupport_no_dtype_for_scalars(self) -> None:
        dim = IntVar([2, 128])
        X1 = gen_input_tensor([dim, 4], name="X1", dtype="bool")
        X2 = 2
        X3 = 2
        with self.assertRaisesRegex(
            AssertionError, "dtype needs to be provided for scalars"
        ):
            ops.where()(X1, X2, X3)

    def test_unsupport_tensor_of_different_dtype(self) -> None:
        X1 = gen_input_tensor([4, 4], name="X1", dtype="bool")
        X2 = gen_input_tensor([4, 4], name="X2", dtype="float32")
        X3 = gen_input_tensor([4, 4], name="X3", dtype="float64")
        with self.assertRaisesRegex(AssertionError, "Expect tensor of the same dtype"):
            ops.where()(X1, X2, X3)

    def test_dtype_for_scalars(self) -> None:
        dim = IntVar([2, 128])
        X1 = gen_input_tensor([dim, 4], name="X1", dtype="bool")
        X2 = 2
        X3 = 2
        Y = ops.where()(X1, X2, X3, dtype="float32")
        self.assertEqual(Y.dtype(), "float32")

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [param("float16", 3), param("float16", 2)],
                TestEnv.CUDA_SM80: [
                    param("bfloat16", 3),
                    param("bfloat16", 2),
                    param("float32", 8),
                    param("float", 1),
                    param("float", 3),
                ],
                TestEnv.ROCM: [param("float16", 3), param("float16", 2)],
            }
        )
    )
    def test_where(self, dtype: str, M: int) -> None:
        dim = IntVar([2, 3, 128])
        X1 = gen_input_tensor([dim, M], name="X1", dtype="bool")
        X2 = gen_input_tensor([dim, M], name="X2", dtype=dtype)
        X3 = gen_input_tensor([dim, M], name="X3", dtype=dtype)
        Y = ops.where()(X1, X2, X3)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", f"test_where_{self._test_id}")
        self._test_id += 1
        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype) < 0
            x2_pt = get_random_torch_tensor([batch, M], dtype)
            x3_pt = get_random_torch_tensor([batch, M], dtype)
            y_pt = torch.where(x1_pt, x2_pt, x3_pt)
            y = get_torch_empty_tensor([batch, M], dtype)
            inputs = {"X1": x1_pt, "X2": x2_pt, "X3": x3_pt}
            model.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y)

    def test_input_tensor_constant(self) -> None:
        dim = IntVar([2, 3, 128])
        dtype = "float"
        M = 4
        X1 = gen_input_tensor([dim, M], name="X1", dtype="bool")
        X2 = 2
        X3 = gen_input_tensor([dim, M], name="X3", dtype=dtype)
        Y = ops.where()(X1, X2, X3)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", "test_input_tensor_constant")

        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype) < 0
            x2_pt = 2
            x3_pt = get_random_torch_tensor([batch, M], dtype)
            y_pt = torch.where(x1_pt, x2_pt, x3_pt)
            y = get_torch_empty_tensor([batch, M], dtype)
            inputs = {"X1": x1_pt, "X3": x3_pt}
            model.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y)

    def test_other_tensor_constant(self) -> None:
        dim = IntVar([2, 3, 128])
        dtype = "float"
        M = 4
        X1 = gen_input_tensor([dim, M], name="X1", dtype="bool")
        X2 = gen_input_tensor([dim, M], name="X2", dtype=dtype)
        X3 = 2
        Y = ops.where()(X1, X2, X3)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", "test_other_tensor_constant")

        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype) < 0
            x2_pt = get_random_torch_tensor([batch, M], dtype)
            x3_pt = 2
            y_pt = torch.where(x1_pt, x2_pt, x3_pt)
            y = get_torch_empty_tensor([batch, M], dtype)
            inputs = {"X1": x1_pt, "X2": x2_pt}
            model.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y)

    def test_both_tensors_constant(self) -> None:
        dim = IntVar([2, 3, 128])
        dtype = "float"
        M = 4
        X1 = gen_input_tensor([dim, M], name="X1", dtype="bool")
        X2 = 4
        X3 = 2
        Y = ops.where()(X1, X2, X3, dtype=dtype)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", "test_both_tensors_constant")

        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype) < 0
            x2_pt = 4
            x3_pt = 2
            y_pt = torch.where(x1_pt, x2_pt, x3_pt).to(torch.float32)
            y = get_torch_empty_tensor([batch, M], dtype)
            inputs = {"X1": x1_pt}
            model.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y)

    def test_integration_with_relational(self) -> None:
        dim = IntVar([2, 3, 128])
        dtype = "float"
        M = 4
        X1 = gen_input_tensor([dim, M], name="X1", dtype=dtype)
        X2 = gen_input_tensor([dim, M], name="X2", dtype=dtype)
        X3 = ops.ge()(X1, X2)
        Y = ops.where()(X3, X1, X2)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = detect_target()
        model = compile_model(Y, target, "./tmp", "test_integration_with_relational")
        for batch in dim._attrs["values"]:
            x1_pt = get_random_torch_tensor([batch, M], dtype)
            x2_pt = get_random_torch_tensor([batch, M], dtype)
            x3_pt = torch.ge(x1_pt, x2_pt)
            y_pt = torch.where(x3_pt, x1_pt, x2_pt)
            y = get_torch_empty_tensor([batch, M], dtype)
            inputs = {"X1": x1_pt, "X2": x2_pt}
            model.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
