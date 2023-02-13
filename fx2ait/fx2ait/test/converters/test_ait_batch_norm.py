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


class TestBatchNormConverter(AITTestCase):
    def test_batch_norm1d(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs1 = [torch.randn(5, 3).cuda().half()]
        self.run_test(
            model,
            inputs1,
            expected_ops={acc_ops.batch_norm},
        )

        inputs2 = [torch.randn(5, 3, 234).cuda().half()]
        self.run_test(
            model,
            inputs2,
            expected_ops={acc_ops.batch_norm},
        )

    def test_batch_norm2d(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 3, 244, 244).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.batch_norm},
        )

    def test_batch_norm3d(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm3d(6)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs = [torch.randn(4, 6, 24, 24, 11).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.batch_norm},
        )
