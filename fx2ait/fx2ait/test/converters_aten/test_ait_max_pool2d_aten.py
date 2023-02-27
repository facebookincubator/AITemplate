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
from parameterized import parameterized


class TestMaxPool2dConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (1, 1, 0),
            ((2, 2), 2, 1),
            ((4, 4), (4, 4), 0),
        ]
    )
    def test_maxpool2d(self, kernel_size, stride, padding):
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
            expected_ops={torch.ops.aten.max_pool2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    @parameterized.expand(
        [
            (1, 1, 0),
            ((2, 2), 2, 1),
            ((4, 4), (4, 4), 0),
        ]
    )
    def test_dynamic_maxpool2d(self, kernel_size, stride, padding):
        class TestModule(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding)

            def forward(self, x):
                return self.pool(x)

        model = TestModule(kernel_size, stride, padding).half().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 4, 224, 224],
            ],
            inputs_max=[
                [10, 4, 256, 256],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.max_pool2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
