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
import torch.nn.functional as F
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized


class TestAitPoolingConverter(AITTestCase):
    @parameterized.expand(
        [
            "avg_pool2d",
            "max_pool2d",
        ]
    )
    def test_pooling2d_with_default_inputs(self, opname):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fn = getattr(F, opname)

            def forward(self, x):
                return self.fn(x, 2)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 4, 256, 256).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={getattr(acc_ops, opname)},
        )

    @parameterized.expand(
        [
            "max_pool3d",
        ]
    )
    def test_pooling3d_with_default_inputs(self, opname):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fn = getattr(F, opname)

            def forward(self, x):
                return self.fn(x, 1)

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 4, 8, 256, 256).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={getattr(acc_ops, opname)},
        )
