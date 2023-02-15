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
from parameterized import parameterized


class TestMaxPool2dConverter(AITTestCase):
    @parameterized.expand(
        [
            (1, 1, 0),
            ((2, 2), 2, 1),
            ((4, 4), (4, 4), 0),
        ]
    )
    def test_avgpool2d(self, kernel_size, stride, padding):
        class TestModule(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding)

            def forward(self, x):
                return self.pool(x)

        model = TestModule(kernel_size, stride, padding).half().cuda()
        inputs = [torch.randn(1, 4, 256, 256).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.max_pool2d},
        )
