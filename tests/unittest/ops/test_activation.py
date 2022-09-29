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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FusedElementwiseTestCase(unittest.TestCase):
    def _test_leaky_relu(self, input_size, negative_slope=0.01, test_name="leaky_relu"):
        assert len(input_size) == 2
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        slope = Tensor(
            shape=[],
            dtype="float16",
            name="slope",
            value=negative_slope,
        )
        X2 = ops.elementwise(FuncEnum.LRELU)(X1, slope)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().half()
        OP_pt = torch.nn.LeakyReLU(negative_slope)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_relu(self, input_size, test_name="relu"):
        assert len(input_size) == 2
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.RELU)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().half()
        x2_pt = torch.relu(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def _test_hardtanh(self, input_size, min_val=-1, max_val=1, test_name="hard_tanh"):
        assert len(input_size) == 2
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X_min = Tensor(
            shape=[],
            dtype="float16",
            name="min_val",
            value=min_val,
            is_input=True,
        )
        X_max = Tensor(
            shape=[],
            dtype="float16",
            name="max_val",
            value=max_val,
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.HARDTANH)(X1, X_min, X_max)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().half()
        OP_pt = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
        x2_pt = OP_pt(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_lrelu(self):
        self._test_leaky_relu([512, 512], test_name="leaky_relu_1")
        self._test_leaky_relu(
            [1024, 1024], negative_slope=0.5, test_name="leaky_relu_2"
        )

    def test_htanh(self):
        self._test_hardtanh([512, 512], test_name="hard_tanh_1")
        self._test_hardtanh(
            [1024, 1024], min_val=-2, max_val=2, test_name="hard_tanh_2"
        )

    def test_relu(self):
        self._test_relu([512, 512], test_name="relu_1")


if __name__ == "__main__":
    unittest.main()
