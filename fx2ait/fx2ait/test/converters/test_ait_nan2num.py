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


class TestNan2NumConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default"),
            param("nan", nan=1.0),
            param("posinf", posinf=1.0),
            param("neginf", neginf=-1.0),
        ]
    )
    def test_nan_to_num(self, name, nan=None, posinf=None, neginf=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

        model = TestModule().cuda().half()
        inputs = [
            torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
            .half()
            .cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.nan_to_num})
