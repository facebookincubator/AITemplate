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


class TestMaskedSelectConverter(AITTestCase):
    @parameterized.expand(
        [
            param("random", torch.randn(5, 10), torch.randn(5, 10)),
            param("all_neg", torch.zeros(5, 10), torch.ones(5, 10)),
            param("all_pos", torch.ones(5, 10), torch.zeros(5, 10)),
        ]
    )
    def test_masked_select(self, _, a, b):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return torch.masked_select(input=x, mask=mask)

        model = TestModule().eval().half().cuda()
        boolTensor = a > b

        inputs = [torch.randn(5, 10).half().cuda(), boolTensor.cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.masked_select})
