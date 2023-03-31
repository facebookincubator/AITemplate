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


class TestFullConverter(AITTestCase):
    def test_new_full(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                full = x.new_full((2, 6), 2.2)
                return torch.cat([full, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.new_full})

    def test_full_like(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                full = torch.full_like(x, 2.2)
                return torch.cat([full, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.full_like})

    def test_new_ones(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                full = x.new_ones((2, 6))
                return torch.cat([full, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.cat, acc_ops.new_ones})

    def test_ones_like(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                ones = torch.ones_like(x)
                return torch.cat([ones, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.cat, acc_ops.ones_like})

    def test_new_zeros(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                zeros = x.new_zeros((2, 6))
                return torch.cat([zeros, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.cat, acc_ops.new_zeros})

    def test_zeros_like(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                zeros = torch.zeros_like(x)
                return torch.cat([zeros, x], dim=1)

        model = TestModule().cuda().half()
        input = [torch.randn([2, 3]).cuda().half()]
        self.run_test(model, input, expected_ops={acc_ops.zeros_like})
