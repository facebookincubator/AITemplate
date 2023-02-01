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


class TestGeluConverter(AITTestCase):
    def test_gelu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x)

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})

    def test_fast_gelu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x, approximate="tanh")

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})

    @parameterized.expand(
        [
            ("none"),
            ("tanh"),
        ]
    )
    def test_gelu_module(self, name):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU(approximate=name)

            def forward(self, x):
                return self.gelu(x)

        inputs = [torch.randn(3, 10, 20).cuda().half()]
        model = TestModule().cuda().half()

        self.run_test(model, inputs, expected_ops={acc_ops.gelu})
