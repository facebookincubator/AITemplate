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
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestLeakyReluConverter(AITTestCase):
    def test_leaky_relu(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.leaky_relu(x, negative_slope=0.05)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.leaky_relu})
