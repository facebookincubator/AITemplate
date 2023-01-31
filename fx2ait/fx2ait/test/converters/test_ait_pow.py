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


class TestPowConverter(AITTestCase):
    @parameterized.expand([("int", 3), ("float", 0.25)])
    def test_pow(self, _, exp):
        class Pow(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.pow(x, exp)

        model = Pow().half().cuda()
        input = [torch.randn(3, 3).half().cuda()]
        self.run_test(model, input, expected_ops={acc_ops.pow})
