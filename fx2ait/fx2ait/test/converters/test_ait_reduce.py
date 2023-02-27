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
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestSumConverter(AITTestCase):
    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
        ]
    )
    def test_sum(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sum(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.sum})

    def test_sum_no_dim(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.sum(x)

        model = TestModule().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.sum})

    @parameterized.expand(
        [
            ["default", None, False],
            ["specified_dims", (0, 1, 2), False],
        ]
    )
    def test_sum_multi_dims(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return y + torch.sum(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(2, 3, 5).half().cuda()] * 2
        self.run_test(model, inputs, expected_ops={acc_ops.add, acc_ops.sum})


class TestMeanConverter(AITTestCase):
    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
        ]
    )
    def test_mean(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mean(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.mean})

    @parameterized.expand(
        [
            ["none", None, False],
            ["specified_dims", (0, 1, 2), False],
        ]
    )
    def test_mean_multi_dims(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return y + torch.mean(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(2, 3, 5).half().cuda() + 1] * 2
        self.run_test(model, inputs, expected_ops={acc_ops.mean})
