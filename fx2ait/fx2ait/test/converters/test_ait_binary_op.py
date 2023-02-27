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
import operator
from typing import Callable, List, Tuple, Union

import torch

from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized

TWO_TENSOR_INPUTS = [
    (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
    (torch.randn(3, 4), torch.randn(2, 3, 4)),
    (torch.randn(2, 3, 4), torch.randn(3, 4)),
    (torch.randn(1, 1, 1), torch.randn(2, 3, 4)),
    (torch.randn(2, 3, 4), torch.randn(1)),
    (torch.randn(2, 3, 4), torch.randn(1, 1, 1)),
    (torch.randn(1, 3, 4), torch.randn(5, 1, 4)),
    (torch.randn(1), torch.randn(2, 3, 4)),
]


class TestBinaryOpConverter(AITTestCase):
    @parameterized.expand(
        [
            [
                "add",
                operator.add,
                acc_ops.add,
                TWO_TENSOR_INPUTS,
            ],
            [
                "sub",
                operator.sub,
                acc_ops.sub,
                TWO_TENSOR_INPUTS,
            ],
            [
                "mul",
                operator.mul,
                acc_ops.mul,
                TWO_TENSOR_INPUTS,
            ],
            # Add .clamp() to avoid division by zero
            [
                "div",
                operator.truediv,
                acc_ops.div,
                [(lhs, rhs.clamp(min=0.01)) for lhs, rhs in TWO_TENSOR_INPUTS],
            ],
            # TODO enable full list of test when OSS python version upgrade to include pyhton floordiv fix
            # [
            #     "floor_div",
            #     operator.floordiv,
            #     acc_ops.floor_div,
            #     [
            #         (TWO_TENSOR_INPUTS[i][0], TWO_TENSOR_INPUTS[i][1].clamp(min=0.01))
            #         for i in range(0, 2)
            #     ],
            # ],
        ]
    )
    def test_two_tensors(
        self,
        name: str,
        op: Callable,
        acc_op: Callable,
        inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return op(x, y)

        for lhs, rhs in inputs:
            model = TestModule().cuda()
            lhs = lhs.half().cuda()
            rhs = rhs.half().cuda()
            self.run_test(model, [lhs, rhs], expected_ops={acc_op})

    @parameterized.expand(
        [
            param("add_int", 1, operator.add, acc_ops.add),
            param("add_float", 0.5, operator.add, acc_ops.add),
            param("mul_int", 1, operator.mul, acc_ops.mul),
            param("mul_float", 0.5, operator.mul, acc_ops.mul),
            param("div_int", 1, operator.truediv, acc_ops.div),
            param("div_float", 0.5, operator.truediv, acc_ops.div),
        ]
    )
    def test_scalar_operand(
        self, name: str, scalar: Union[int, float], op: Callable, acc_op: Callable
    ) -> None:
        class TestModuleScalarLhs(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return op(scalar, x)

        class TestModuleScalarRhs(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return op(x, scalar)

        model_scalar_lhs = TestModuleScalarLhs().cuda()
        self.run_test(
            model_scalar_lhs,
            [torch.randn(2, 3, 4).half().cuda()],
            expected_ops={acc_op},
        )

        model_scalar_rhs = TestModuleScalarRhs().cuda()
        self.run_test(
            model_scalar_rhs,
            [torch.randn(2, 3, 4).half().cuda()],
            expected_ops={acc_op},
        )

    @parameterized.expand(
        [
            param("add", 1, 3, operator.add, acc_ops.add),
            param("mul", 0.5, 1, operator.mul, acc_ops.mul),
            param("sub", 1, 0.5, operator.sub, acc_ops.sub),
            param("div", 0.5, 0.5, operator.truediv, acc_ops.div),
        ]
    )
    def test_constant_operand(
        self,
        name: str,
        x: Union[int, float],
        y: Union[int, float],
        op: Callable,
        acc_op: Callable,
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                x = op(input.size()[-1], input.size()[-1])
                return op(x, input)

        model = TestModule().cuda()
        self.run_test(
            model,
            [torch.randn(2, 4).half().cuda()],
            expected_ops={acc_op},
        )

    # This is a common binary op combo usage for ads models.
    def test_binary_op_combo(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                x = input.size()[0] * input.size()[0]
                return torch.reshape(input, [-1, x])

        model = TestModule().cuda()
        self.run_test(
            model,
            [torch.randn(2, 4).half().cuda()],
            expected_ops={acc_ops.reshape, acc_ops.mul},
        )
