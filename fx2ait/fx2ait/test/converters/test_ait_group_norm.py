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


class TestGroupNormTensor(AITTestCase):
    @parameterized.expand(
        [
            [True],
            [False],
        ]
    )
    def test_group_norm(self, affine):
        class GN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = torch.nn.GroupNorm(3, 6, affine=affine)

            def forward(self, x):
                return self.gn(x)

        mod = GN().half().cuda()
        inputs = [torch.randn(2, 6, 4, 5).half().cuda()]
        self.run_test(
            mod,
            inputs,
            expected_ops={},
        )
