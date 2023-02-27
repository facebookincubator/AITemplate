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
import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestMatMulConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 3], [3, 4]],
            [[2, 3, 4], [2, 4, 6]],
            [[2, 3, 4], [4, 6]],
            [[3, 4], [5, 4, 6]],
            [[2, 2, 2, 3, 4], [4, 6]],
        ]
    )
    def test_simple(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    def test_mm(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.mm(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3).half().cuda(),
            torch.randn(3, 4).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    @parameterized.expand(
        [
            [[1, 2, 3], [1, 3, 4]],
            [[3, 2, 3], [3, 3, 4]],
        ]
    )
    def test_bmm(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.bmm(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    @parameterized.expand(
        [
            [[1, 1, 3, 4], [1, 1, 4, 6]],
            [[1, 2, 3, 4], [1, 2, 4, 6]],
            [[4, 1, 3, 4], [4, 1, 4, 6]],
            [[4, 2, 3, 4], [4, 2, 4, 6]],
        ]
    )
    def test_matmul_with_4d_tensors(self, lhs_shape, rhs_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*lhs_shape).half().cuda(),
            torch.randn(*rhs_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.matmul})

    def test_reshape_bmm(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = torch.reshape(x, [-1, 3, 4])
                y = torch.reshape(y, [-1, 4, 6])
                return torch.bmm(x, y)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[[2, 3, 4], [2, 4, 6]],
            inputs_max=[[20, 3, 4], [20, 4, 6]],
            dtype_list=[
                torch.float16,
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={acc_ops.matmul}
        )

    def test_reshape_4d_bmm(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = torch.reshape(x, [-1, 1, 3, 4])
                y = torch.reshape(y, [-1, 1, 4, 6])
                return torch.matmul(x, y)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[[2, 3, 4], [2, 4, 6]],
            inputs_max=[[20, 3, 4], [20, 4, 6]],
            dtype_list=[
                torch.float16,
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={acc_ops.matmul}
        )
