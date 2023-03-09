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
Unittests for special activation Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import filter_test_cases_by_test_env
from aitemplate.utils.serialization.serdes_code import (
    dump_program,
    get_inputs_from_graph,
    get_program,
)


_LOGGER = logging.getLogger(__name__)


class SerDesTestCase(unittest.TestCase):
    def test_get_inputs(self):
        X1 = Tensor(
            shape=[IntImm(3), IntImm(4)],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X3._attrs["is_output"] = True
        X3._attrs["name"] = "output0"

        test_get_inputs_path = "./tmp/test_serdes_get_inputs.py"
        dump_program(X3, test_get_inputs_path)
        inputs = get_inputs_from_graph(test_get_inputs_path)

        self.assertEqual(len(inputs), 2)
        self.assertIsNotNone(inputs.get("input_0", None))
        self.assertIsNotNone(inputs.get("input_1", None))
        for input_ in [X1, X2]:
            shape = input_.shape()
            graph_shape = inputs[input_._attrs["name"]]
            self.assertEqual(len(shape), len(graph_shape))
            for x, y in zip(shape, graph_shape):
                self.assertEqual(x, y)

    def test_simple_serdes(self):
        X1 = Tensor(
            shape=[IntImm(3), IntImm(4)],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(1)],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X3._attrs["is_output"] = True
        X3._attrs["name"] = "output0"

        test_path = "./tmp/test_simple_serdes.py"
        dump_program(X3, test_path)

        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "simple_serdes")

        x1_pt = torch.randn(3, 4).cuda().half()
        x2_pt = torch.randn(1).cuda().half()
        x3_pt = x1_pt + x2_pt

        x3 = torch.empty_like(x3_pt)
        module.run_with_tensors({"input_0": x1_pt, "input_1": x2_pt}, {"output0": x3})
        self.assertTrue(torch.allclose(x3, x3_pt, atol=1e-2, rtol=1e-2))

    def test_multi_outputs(self):
        X_pt = torch.randn(8, 10).cuda().half()
        Ys_pt = torch.split(X_pt, 4)

        X = Tensor(shape=[8, 10], dtype="float16", name="input_0", is_input=True)
        Ys = ops.split()(X, 4)

        self.assertEqual(len(Ys_pt), len(Ys))

        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True

        test_path = "./tmp/test_serdes_multi_outputs.py"
        dump_program(Ys, test_path)

        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "serdes_multi_outputs")

        y_shapes = [(4, 10), (4, 10)]
        outputs = {
            f"output_{idx}": torch.empty(y_shape).cuda().half()
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors({"input_0": X_pt}, outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))


class SerDesSpecialOpTestCase(unittest.TestCase):
    def test_elementwise(self):
        X1 = Tensor(
            shape=[IntImm(3), IntImm(4)],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.MIN)(X1, 0.5)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        test_path = "./tmp/test_serdes_elementwise.py"
        dump_program(X2, test_path)
        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "serdes_elementwise")

        x1_pt = torch.randn(3, 4).cuda().half()
        x2_pt = torch.clamp(x1_pt, max=0.5)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors({"input_0": x1_pt}, {"output0": x2})
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_concat(self):
        X_pts = [torch.randn(8, 10).cuda().half() for _ in range(5)]
        Y_pt = torch.cat(X_pts)

        Xs = [
            Tensor(shape=[8, 10], dtype="float16", name=f"input_{i}", is_input=True)
            for i in range(5)
        ]
        Y = ops.concatenate()(Xs)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        test_path = "./tmp/test_serdes_concat.py"
        dump_program(Y, test_path)

        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "serdes_concat")

        input_tensors_ait = {f"input_{idx}": X_pts[idx] for idx in range(5)}
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(input_tensors_ait, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_reshape(self):
        X = Tensor(
            shape=[5, 4, 6, 8],  # 960 total
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        Y = ops.reshape()(X, [-1, 6, 32])
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        test_path = "./tmp/test_serdes_reshape.py"
        dump_program(Y, test_path)

        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "serdes_reshape")

        X_pt = torch.randn(5, 4, 6, 8).cuda().half()
        Y_pt = torch.reshape(X_pt, (-1, 6, 32))
        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_group_gemm_rcr_sm80(self):
        target = detect_target()
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64

        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        OP = ops.group_gemm_rcr()
        Y1, Y2 = OP(operand_groups=[[X1, W1], [X2, W2]])
        Y1._attrs["name"] = "y1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "y2"
        Y2._attrs["is_output"] = True

        test_path = "./tmp/test_serdes_group_gemm.py"
        dump_program([Y1, Y2], test_path)
        outputs, _ = get_program(test_path)
        module = compile_model(outputs, target, "./tmp", "serdes_group_gemm")

        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
        }
        y1 = torch.empty([M, N1]).cuda().half()
        y2 = torch.empty([M, N2]).cuda().half()
        outputs = {"y1": y1, "y2": y2}

        module.run_with_tensors(inputs, outputs)
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y2_pt, y2, atol=1e-1, rtol=1e-1))

    def test_dynamic_slice(self):
        batch_sizes = [5, 6, 7]
        input_shape = [2, 3, 4]
        X = Tensor(
            shape=[IntVar(values=batch_sizes, name="input_batch_0"), *input_shape],
            name="input_0",
            is_input=True,
        )
        start_indices = [2, 1, 0, 1]
        end_indices = [5, 2, 2, 4]
        Y = ops.dynamic_slice()(X, start_indices=start_indices, end_indices=end_indices)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        test_path = "./tmp/test_serdes_dynamic_slice.py"
        dump_program(Y, test_path)
        outputs, _ = get_program(test_path)

        target = detect_target()
        module = compile_model(outputs, target, "./tmp", "serdes_dynamic_slice")

        for batch in batch_sizes:
            # generate torch reference result
            X_pt = torch.randn(batch, *input_shape).cuda().half()

            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            y_pt = Y_pt.cpu().numpy()

            y = torch.empty(y_pt.shape).cuda().half()
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))


filter_test_cases_by_test_env(SerDesTestCase)
filter_test_cases_by_test_env(SerDesSpecialOpTestCase)

if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
