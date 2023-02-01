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


class TestSqueezeConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", dim=None, shape=[2, 1, 1, 3]),
            param("1", dim=1, shape=[2, 1, 1, 3]),
            param("-1", dim=-1, shape=[2, 1, 3, 1]),
        ]
    )
    def test_squeeze(self, name, dim, shape):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                squeeze = (
                    torch.squeeze(y, dim=dim) if dim is not None else torch.squeeze(y)
                )
                return squeeze

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.squeeze})
