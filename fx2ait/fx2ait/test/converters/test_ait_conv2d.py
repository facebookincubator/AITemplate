# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestConv2dConverter(AITTestCase):
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
        inputs = [torch.randn(1, 3, 224, 224).cuda().half()]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.conv2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
