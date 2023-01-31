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
from typing import Callable, List, Tuple

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


TWO_TENSOR_INPUTS = [
    (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
    (torch.randn(3, 4), torch.randn(2, 3, 4)),
    (torch.randn(2, 3, 4), torch.randn(3, 4)),
    (torch.randn(1, 1, 1), torch.randn(2, 3, 4)),
    (torch.randn(1), torch.randn(2, 3, 4)),
    (torch.randn(2, 3, 4), torch.randn(1)),
    (torch.randn(2, 3, 4), torch.randn(1, 1, 1)),
    (torch.randn(1, 3, 4), torch.randn(5, 1, 4)),
]


class TestATenBinaryOpConverter(DispatchTestCase):
    @parameterized.expand(
        [
            [
                "add",
                operator.add,
                torch.ops.aten.add.Tensor,
                TWO_TENSOR_INPUTS,
            ],
            [
                "sub",
                operator.sub,
                torch.ops.aten.sub.Tensor,
                TWO_TENSOR_INPUTS,
            ],
            [
                "mul",
                operator.mul,
                torch.ops.aten.mul.Tensor,
                TWO_TENSOR_INPUTS,
            ],
            [
                "div",
                operator.truediv,
                torch.ops.aten.div.Tensor,
                [(lhs, rhs.clamp(min=0.01)) for lhs, rhs in TWO_TENSOR_INPUTS],
            ],
        ]
    )
    def test_two_tensors(
        self,
        name: str,
        op: Callable,
        aten_op: Callable,
        inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return op(x, y)

        for lhs, rhs in inputs:
            model = TestModule().cuda()
            lhs = lhs.half().cuda()
            rhs = rhs.half().cuda()
            self.run_test(model, [lhs, rhs], expected_ops={aten_op})

    @parameterized.expand(
        [
            [
                "dynamic_add",
                operator.add,
                torch.ops.aten.add.Tensor,
            ],
            [
                "dynamic_sub",
                operator.sub,
                torch.ops.aten.sub.Tensor,
            ],
            [
                "dynamic_sub",
                operator.mul,
                torch.ops.aten.mul.Tensor,
            ],
            [
                "dynamic_div",
                operator.truediv,
                torch.ops.aten.div.Tensor,
            ],
        ]
    )
    def test_dynamic_two_tensors(
        self,
        name: str,
        op: Callable,
        aten_op: Callable,
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return op(x, y)

        m = TensorSpec.gen_int_var_min_max(1, 32, "dynamic_m")
        n = TensorSpec.gen_int_var_min_max(3, 1024, "dynamic_n")
        k = TensorSpec.gen_int_var_min_max(4, 2048, "dynamic_k")
        model = TestModule().cuda().half()
        # AIT can automatically calculate broadcast
        input_spec = TensorSpec.create_spec_from_int_vars(
            [[m, n, k], [n, k]], dtype_list=[torch.float16] * 2
        )

        self.run_test_with_dynamic_shape(
            model,
            input_spec,
            expected_ops={aten_op},
        )
