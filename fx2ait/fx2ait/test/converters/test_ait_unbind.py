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
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized
from torch import nn


class TestUnbindTensor(AITTestCase):
    @parameterized.expand(
        [
            ("positive_dim", 2),
            ("negative_dim", -1),
        ]
    )
    def test_unbind(self, name, dim):
        class GetItem(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.unbind(x, dim=dim)
                z = y[0]
                return z

        mod = GetItem().half().cuda()
        inputs = [torch.randn(2, 3, 4).half().cuda()]
        self.run_test(
            mod,
            inputs,
            expected_ops={},
        )
