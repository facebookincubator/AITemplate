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
import torch
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestConvtTranspose2dConverter(DispatchTestCase):
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
            expected_ops={torch.ops.aten.convolution.default},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )

    # # only works when in_ch == out_ch
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
            expected_ops={torch.ops.aten.convolution.default},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=[0, 3, 1, 2],
        )
