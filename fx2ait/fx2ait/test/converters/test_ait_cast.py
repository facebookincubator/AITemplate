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
import unittest

import torch
from aitemplate.testing import detect_target
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized
from torch import nn


@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class TestCastConverter(AITTestCase):
    @parameterized.expand(
        [
            ("half_to_float", torch.half, torch.float),
            ("float_to_half", torch.float, torch.half),
            ("half_to_bf16", torch.half, torch.bfloat16),
            ("bool_to_half", torch.bool, torch.half),
        ]
    )
    def test_cast(
        self,
        name,
        dtype,
        cast_dtype,
    ):
        class Cast(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.cast_ty = dtype

            def forward(self, x):
                y = x.to(self.cast_ty)
                return x.to(y.dtype) + y

        model = Cast(cast_dtype).cuda()
        x = torch.randn(3, 4, 5)
        if dtype == torch.bool:
            x = x < 0.5
        else:
            x.to(dtype)
        inputs = [x]
        self.run_test(model, inputs, expected_ops={acc_ops.to_dtype})
