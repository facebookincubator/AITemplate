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

import logging

import torch
from aitemplate.testing.test_utils import filter_test_cases_by_params, TestEnv
from aitemplate.utils.torch_utils import string_to_torch_dtype
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase, torch_type_to_lower_precision
from parameterized import parameterized


class TestConv2dConverter(AITTestCase):
    def _test_conv2d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        in_channel=3,
        bias=True,
        ait_dtype="float16",
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channel, 36, kernel_size, stride, padding, dilation, groups, bias
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        logging.info(f"Running test {test_name}.")

        dtype = string_to_torch_dtype(ait_dtype)
        model = TestModule().cuda().to(dtype)
        inputs = [torch.randn(1, in_channel, 224, 224).cuda().to(dtype)]
        self.run_test(
            model,
            inputs,
            expected_ops={acc_ops.conv2d},
            precision=torch_type_to_lower_precision(dtype),
        )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv2d(self, ait_dtype):
        self._test_conv2d(f"{ait_dtype}_default", 1, ait_dtype=ait_dtype)
        self._test_conv2d(f"{ait_dtype}_no_bias", 1, bias=False, ait_dtype=ait_dtype)
        self._test_conv2d(
            f"{ait_dtype}_tuple_parameters", 1, (1, 1), (1, 1), ait_dtype=ait_dtype
        )
        self._test_conv2d(
            f"{ait_dtype}_non_zero_padding", 1, padding=1, ait_dtype=ait_dtype
        )
        self._test_conv2d(
            f"{ait_dtype}_non_unary_params",
            3,
            2,
            padding=1,
            bias=False,
            ait_dtype=ait_dtype,
        )
        self._test_conv2d(f"{ait_dtype}_dilation", 1, dilation=2, ait_dtype=ait_dtype)
        self._test_conv2d(
            f"{ait_dtype}_multi_group", 1, 1, 1, 1, 3, bias=True, ait_dtype=ait_dtype
        )
        self._test_conv2d(
            f"{ait_dtype}_padding_3", 1, in_channel=3, ait_dtype=ait_dtype
        )
        self._test_conv2d(
            f"{ait_dtype}_padding_7", 1, in_channel=7, ait_dtype=ait_dtype
        )
