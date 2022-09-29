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
Unittests for fused_elementwise Operator.
"""
import math
import unittest
from typing import List

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops, transform
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


class FusedElementwiseTestCase(unittest.TestCase):
    def test_fused_elementwise_constructor(self):
        BATCH_SIZE = 1024
        M = 256
        K = 128

        op1 = ops.elementwise(FuncEnum.ADD)
        op1._attrs["name"] = "e1"
        op2 = ops.elementwise(FuncEnum.TANH)
        op2._attrs["name"] = "e2"
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype="float16",
            name="X2",
            value=3.0,
        )
        X3 = op1(X1, X2)
        X3._attrs["name"] = "X3"
        X4 = op2(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        graph = transform.toposort(X4)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        transform.refine_graph(graph)

        fused_op = ops.fused_elementwise([op1, op2])
        fused_op._attrs["name"] = "fused_elementwise0"

        self.assertEqual(fused_op._attrs["inputs"], [X1])
        self.assertEqual(fused_op._attrs["outputs"], [X4])

        self.assertEqual(X4._attrs["src_ops"], {fused_op})
        self.assertEqual(X1._attrs["dst_ops"], {fused_op})

        self.assertEqual(fused_op._attrs["depth"], 0)
        self.assertEqual(X1._attrs["depth"], 0)
        self.assertEqual(X4._attrs["depth"], 2)

    def _test_fused_elementwise_e2e(self, batch_sizes, ms, ks, test_name):
        X1 = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(batch_sizes),
                shape_utils.gen_int_var_min_max(ms),
                shape_utils.gen_int_var_min_max(ks),
            ],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype="float16",
            name="X2",
            value=3.0,
        )
        X3 = X1 + X2
        X3._attrs["name"] = "X3"
        X4 = ops.tanh(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            X4,
            target,
            "./tmp",
            "fused_elementwise_{}".format(test_name),
        )

        for batch_size in batch_sizes:
            for m in ms:
                for k in ks:
                    x1_pt = torch.randn(batch_size, m, k).cuda().half()
                    x4_pt = torch.tanh(x1_pt + 3.0)

                    x4 = torch.empty([batch_size, m, k]).cuda().half()
                    module.run_with_tensors([x1_pt], [x4])
                    self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_e2e(self):
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024], ms=[256], ks=[128], test_name="static_shapes"
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1, 99, 998, 1024],
            ms=[256],
            ks=[128],
            test_name="dynamic_batch_size",
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024], ms=[1, 128, 256], ks=[128], test_name="dynamic_m"
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024], ms=[256], ks=[1, 3, 8, 128], test_name="dynamic_k"
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[700, 80, 1024],
            ms=[23, 78, 256],
            ks=[10, 30, 128],
            test_name="dynamic_all",
        )

    def test_fused_elementwise_kernel1(self):
        BATCH_SIZE = 1024
        M = 1496

        X1 = Tensor(
            shape=[IntImm(BATCH_SIZE), IntImm(2), IntImm(M)],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype="float16",
            name="constant_number",
            value=1.0,
        )
        X3 = Tensor(
            shape=[IntImm(2), IntImm(M)],
            dtype="float16",
            name="constant_matrix",
            is_input=True,
        )
        X4 = ops.elementwise(FuncEnum.SIGN)(X1)
        X5 = ops.elementwise(FuncEnum.ABS)(X1)
        X6 = ops.elementwise(FuncEnum.ADD)(X5, X2)
        X7 = ops.elementwise(FuncEnum.LOGE)(X6)
        X8 = ops.elementwise(FuncEnum.MUL)(X4, X7)
        X9 = ops.elementwise(FuncEnum.MUL)(X8, X3)
        X9._attrs["is_output"] = True
        X9._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X9, target, "./tmp", "fused_elementwise_kernel1")

        x1_pt = torch.randn(BATCH_SIZE, 2, M).cuda().half()
        x3_pt = torch.randn(2, M).cuda().half()
        x9_pt = torch.sign(x1_pt) * torch.log1p(torch.abs(x1_pt)) * x3_pt

        inputs = {"input0": x1_pt, "constant_matrix": x3_pt}
        x9 = torch.empty([BATCH_SIZE, 2, M]).cuda().half()
        module.run_with_tensors(inputs, [x9])
        self.assertTrue(torch.allclose(x9, x9_pt, atol=1e-2, rtol=1e-2))

    def _test_sigmoid(self, input_size, test_name="sigmoid"):
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.SIGMOID)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().half()
        x2_pt = torch.sigmoid(x1_pt)

        x2 = torch.empty(input_size).cuda().half()
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_sigmoid(self):
        self._test_sigmoid([1024, 2 * 1496], "sigmoid_1")
        self._test_sigmoid([1024, 23744], "sigmoid_2")
        self._test_sigmoid([1024, 70144], "sigmoid_3")

    def _test_tanh(self, input_size, test_name="tanh"):
        assert len(input_size) == 2
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.TANH)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().half()
        x2_pt = torch.tanh(x1_pt)

        x2 = torch.empty(input_size).cuda().half()
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_tanh(self):
        self._test_tanh([1024, 22400], "tanh_1")
        self._test_tanh([1024, 70144], "tanh_2")
        self._test_tanh([1024, 23744], "tanh_3")

    def _test_min_max(
        self, input_size: List[List[int]], test_name: str, is_min: bool, add_nans: bool
    ) -> None:
        assert len(input_size) == 2
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input1",
            is_input=True,
        )
        if is_min:
            result = ops.elementwise(FuncEnum.MIN)(X0, X1)
        else:
            result = ops.elementwise(FuncEnum.MAX)(X0, X1)

        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = torch.randn(input_size).cuda().half()
        x1_pt = torch.randn(input_size).cuda().half()
        if add_nans:
            x1_pt[0].fill_(float("nan"))

        if is_min:
            x2_pt = torch.min(x0_pt, x1_pt)
        else:
            x2_pt = torch.max(x0_pt, x1_pt)
        x2_np = x2_pt.cpu().numpy()

        inputs = {"input0": x0_pt, "input1": x1_pt}
        x2 = torch.empty(input_size).cuda().half()
        module.run_with_tensors(inputs, [x2])
        x2 = x2.cpu().numpy()

        if add_nans:
            nans = np.full(x2_np[0].shape, np.nan)
            np.testing.assert_allclose(nans, x2_np[0], equal_nan=True)
            np.testing.assert_allclose(nans, x2[0], equal_nan=True)

        np.testing.assert_allclose(x2, x2_np, atol=1e-2, rtol=1e-2)

    def test_min(self):
        self._test_min_max([512, 512], test_name="min_1", is_min=True, add_nans=False)
        self._test_min_max([512, 512], test_name="min_2", is_min=True, add_nans=True)

    def test_max(self):
        self._test_min_max([512, 512], test_name="max_1", is_min=False, add_nans=False)
        self._test_min_max([512, 512], test_name="max_2", is_min=False, add_nans=True)

    def _test_clamp(
        self, input_size: List[List[int]], min_val: int, max_val: int, test_name: str
    ) -> None:
        assert len(input_size) == 2
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        result = ops.clamp()(X0, min_val, max_val)
        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = torch.randn(input_size).cuda().half()

        x1_pt = torch.clamp(x0_pt, min_val, max_val)

        x1 = torch.empty(input_size).cuda().half()
        module.run_with_tensors([x0_pt], [x1])

        self.assertTrue(torch.allclose(x1, x1_pt, atol=1e-2, rtol=1e-2))

    def test_clamp(self):
        self._test_clamp([512, 106], -1, 1, "clamp_0")
        self._test_clamp([128, 46], None, 1, "clamp_1")
        self._test_clamp([56, 265], -1, None, "clamp_2")
        self._test_clamp([17, 123], 1, -1, "clamp_3")

    def test_operator_overload(self):
        input_size = [4, 2]
        X1 = Tensor(
            shape=input_size,
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=input_size,
            dtype="float16",
            name="input1",
            is_input=True,
        )
        OUTPUT = -ops.tanh(X1 + X2) + ops.tanh(X2) + ops.tanh(X1)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", "test_op_overload")

        x1_pt = torch.randn(input_size).cuda().half()
        x2_pt = torch.randn(input_size).cuda().half()
        output_pt = -torch.tanh(x1_pt + x2_pt) + torch.tanh(x2_pt) + torch.tanh(x1_pt)

        output = torch.empty(input_size).cuda().half()
        module.run_with_tensors([x1_pt, x2_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))

    def test_operator_overload_with_constant_number(self):
        input_size = [4, 2]
        X1 = Tensor(
            shape=input_size,
            dtype="float16",
            name="input0",
            is_input=True,
        )
        OUTPUT = 10 / ops.tanh(X1 + 5) - ops.cos(10)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", "test_op_overload")

        x1_pt = torch.randn(input_size).cuda().half()
        output_pt = 10 / torch.tanh(x1_pt + 5) - math.cos(10)
        output = torch.empty(input_size).cuda().half()
        module.run_with_tensors([x1_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
