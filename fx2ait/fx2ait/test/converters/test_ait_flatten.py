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


class TestFlattenConverter(AITTestCase):
    @parameterized.expand(
        [param("default"), param("start", start_dim=1), param("end", end_dim=2)]
    )
    def test_clamp(self, name, start_dim=0, end_dim=-1):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.flatten})
