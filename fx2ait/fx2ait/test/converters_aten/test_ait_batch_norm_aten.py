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
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase


class TestAdaptiveAvgPool2dConverter(DispatchTestCase):
    def test_batch_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                y = self.bn(x)
                y = y.mul(1)
                return y

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 3, 244, 244).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    def test_batch_norm_2layers(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                y = self.bn(x)
                y = y.mul(1)
                y = self.bn2(y)
                return y

        model = TestModule().half().cuda()
        inputs = [torch.randn(1, 3, 244, 244).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    def test_dynamic_batch_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        model = TestModule().half().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 3, 244, 244],
            ],
            inputs_max=[
                [10, 3, 256, 256],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.batch_norm},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
