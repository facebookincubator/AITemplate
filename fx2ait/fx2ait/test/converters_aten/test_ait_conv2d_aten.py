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
from aitemplate.compiler.public import DynamicProfileStrategy
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestConv2dConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", 1),
            param("no_bias", 1, bias=False),
            param("tuple_parameters", 1, (1, 1), (1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("non_unary_params", 3, 2, padding=1, bias=False),
            param("dilation", 1, dilation=2),
            param("multi_group", 1, 1, 1, 1, 3, bias=True),
        ]
    )
    def test_conv2d(
        self,
        name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 36, kernel_size, stride, padding, dilation, groups, bias
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = TestModule().cuda().half()
        inputs = [torch.randn(1, 3, 24, 24).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={
                torch.ops.aten.convolution.default,
                torch.ops.aten.relu.default,
            },
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    @parameterized.expand(
        [
            param("default", 1),
            param("no_bias", 1, bias=False),
            param("tuple_parameters", 1, (1, 1), (1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("non_unary_params", 3, 2, padding=1, bias=False),
            param("dilation", 1, dilation=2),
            param("multi_group", 1, 1, 1, 1, 3, bias=True),
        ]
    )
    def test_dynamic_conv2d(
        self,
        name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 36, kernel_size, stride, padding, dilation, groups, bias
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = TestModule().cuda().half()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 24, 24],
            ],
            inputs_max=[
                [32, 3, 24, 24],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={
                torch.ops.aten.convolution.default,
                torch.ops.aten.relu.default,
            },
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
            # AIT conv2d only support HINTS as dyanmic profiliing strategy.
            dynamic_profile_strategy=DynamicProfileStrategy.HINTS,
        )
