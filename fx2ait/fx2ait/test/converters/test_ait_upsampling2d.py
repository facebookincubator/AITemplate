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
from parameterized import param, parameterized


class TestInterpolateConverter(AITTestCase):
    @parameterized.expand(
        [
            param(scale_factor=1, mode="nearest"),
            param(scale_factor=2, mode="nearest"),
            param(scale_factor=2, mode="bilinear"),
        ]
    )
    def test_interpolate(self, scale_factor, mode):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.interpolate(
                    y, scale_factor=scale_factor, mode=mode
                )
                return x

        model = TestModule().cuda().half()
        inputs = [
            torch.randn([2, 8, 16, 16]).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.interpolate})
