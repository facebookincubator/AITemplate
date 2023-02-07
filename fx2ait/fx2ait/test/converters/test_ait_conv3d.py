# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


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
            permute_inputs=[0, 2, 3, 4, 1],  # inputs should be NDHWC
            permute_outputs=[0, 4, 1, 2, 3],
        )
