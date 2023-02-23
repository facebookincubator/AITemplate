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

import unittest

import torch

from aitemplate.testing import detect_target
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class TestAitConv3d(AITTestCase):
    @parameterized.expand(
        [
            param("conv3d", 3, bias=False),
            param(
                name="conv3d_tuple_parameters",
                kernel_size=3,
                stride=(4, 4, 4),
                padding=(2, 2, 2),
                dilation=2,
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_0",
                kernel_size=3,
                stride=(2, 4, 4),
                padding=(1, 2, 2),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_1",
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 2, 2),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_2",
                kernel_size=3,
                stride=(2, 8, 8),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_3",
                kernel_size=3,
                stride=(1, 4, 4),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_4",
                kernel_size=3,
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_mvit_5",
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=False,
            ),
            param(
                name="conv3d_bias",
                kernel_size=(3, 5, 5),
                stride=(2, 4, 4),
                padding=(1, 2, 2),
                dilation=1,
                ci=8,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=True,
            ),
            param(
                name="conv3d_bias_ndhwc3to8",
                kernel_size=(3, 5, 5),
                stride=(2, 4, 4),
                padding=(1, 2, 2),
                dilation=1,
                ci=3,
                co=96,
                groups=1,
                d=4,
                h=224,
                w=224,
                bias=True,
            ),
        ]
    )
    def test_conv3d(
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
