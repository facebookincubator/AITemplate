# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestConvtTranspose2dConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", 1),
            param("no_bias", 2, bias=False),
            param("tuple_parameters", 1, (1, 1), (1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("non_unary_params", 3, 2, padding=1, bias=False),
        ]
    )
    def test_convtranspose(
        self,
        name,
        kernel_size,
        stride=2,
        padding=0,
        dilation=1,  # only support dilation = 1
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.convtranspose = torch.nn.ConvTranspose2d(
                    192,
                    256,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=0,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.convtranspose(x)

        model = TestModule().cuda().half().eval()
        inputs = [torch.randn(1, 192, 28, 28).cuda().half()]
        _ = model(*inputs)
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.conv_transpose2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    # only works when in_ch == out_ch
    def test_convtranspose_multi_group(
        self,
        name="multi_group",
        kernel_size=2,
        stride=2,
        padding=0,
        dilation=1,  # only support dilation = 1
        groups=2,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.convtranspose = torch.nn.ConvTranspose2d(
                    192,
                    192,  # must to divisblce by 8
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=0,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.convtranspose(x)

        model = TestModule().cuda().half().eval()
        inputs = [torch.randn(1, 192, 28, 28).cuda().half()]
        _ = model(*inputs)
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.conv_transpose2d},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
