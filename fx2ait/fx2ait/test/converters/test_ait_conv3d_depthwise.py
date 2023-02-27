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
from parameterized import param, parameterized


class TestAitDepthwiseConv3d(AITTestCase):
    @parameterized.expand(
        [
            param(
                name="depthwise_conv3d",
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                dilation=1,
                ci=96,
                co=96,
                groups=96,
                d=2,
                h=56,
                w=56,
                bias=True,
            ),
            param(
                name="depthwise_conv3d_2",
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                dilation=1,
                ci=96,
                co=96,
                groups=96,
                d=2,
                h=28,
                w=28,
                bias=True,
            ),
            param(
                name="depthwise_conv3d_3",
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                dilation=1,
                ci=96,
                co=96,
                groups=96,
                d=2,
                h=7,
                w=7,
                bias=True,
            ),
            param(
                name="depthwise_conv3d_4",
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                dilation=1,
                ci=3,
                co=3,
                groups=3,
                d=4,
                h=224,
                w=224,
                bias=True,
            ),
            param(
                name="depthwise_conv3d_no_bias",
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                dilation=1,
                ci=96,
                co=96,
                groups=96,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
        ]
    )
    def test_depthwise_conv3d(
        self,
        name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        ci=8,
        co=8,
        groups=1,
        d=4,
        h=224,
        w=224,
        bias=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    ci,
                    co,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = TestModule().cuda().half()
        inputs = [torch.randn(4, ci, d, h, w).cuda().half()]

        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.conv3d},
        )
