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
import itertools
import math
from typing import Callable, Dict, Set

import torch
from aitemplate.testing.test_utils import filter_test_cases_by_params, TestEnv
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import (
    AITTestCase,
    lower_precision_to_torch_type,
    LowerPrecision,
)
from parameterized import parameterized


unary_ops = [
    (torch.abs, acc_ops.abs),
    (torch.sign, acc_ops.sign),
    (torch.log, acc_ops.log),
    (torch.relu, acc_ops.relu),
    (torch.sin, acc_ops.sin),
    (torch.cos, acc_ops.cos),
    (torch.sqrt, acc_ops.sqrt),
    (torch.clone, acc_ops.clone),
    (torch.neg, acc_ops.neg),
]

TestEnvToPrecision: Dict[TestEnv, Set[LowerPrecision]] = {
    TestEnv.CUDA_LESS_THAN_SM80: [LowerPrecision.FP16, LowerPrecision.FP32],
    TestEnv.CUDA_SM80: [LowerPrecision.BF16],
    TestEnv.ROCM: [LowerPrecision.FP16],
}


class TestUnaryOpsConverter(AITTestCase):
    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                env: [
                    (
                        f"{env}_{op[0].__name__}_{precision.value}",
                        op[0],
                        op[1],
                        precision,
                    )
                    for op, precision in itertools.product(unary_ops, precisions)
                ]
                for env, precisions in TestEnvToPrecision.items()
            }
        )
    )
    def test_unary_ops(
        self, name: str, orig_op: Callable, expected_op, precision: LowerPrecision
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return orig_op(x) * 2.0

        torch_dtype = lower_precision_to_torch_type(precision)
        model = TestModule().cuda().to(torch_dtype)
        inputs = [
            torch.randn(1, 2, 3).cuda().to(torch_dtype),
        ]

        self.run_test(
            model,
            inputs,
            expected_ops={expected_op} if expected_op is not None else {},
            precision=precision,
        )

    def test_sqrt(self):
        class TestModule(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self, y):
                return torch.div(y, math.sqrt(self.x))

        model = TestModule(x=64).cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={})

    def test_to(self):
        class TestModule(torch.nn.Module):
            def forward(self, y):
                return y.to(dtype=torch.float16)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.to_dtype})

    def test_contiguous(self):
        class TestModule(torch.nn.Module):
            def forward(self, y):
                return y.contiguous()

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.contiguous})
