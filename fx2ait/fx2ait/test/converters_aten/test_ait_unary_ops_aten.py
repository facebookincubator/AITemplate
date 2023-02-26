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
from typing import Callable

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


unary_ops = [
    (torch.abs, torch.ops.aten.abs.default),
    (torch.log, torch.ops.aten.log.default),
    (torch.sigmoid, torch.ops.aten.sigmoid.default),
    (torch.sign, torch.ops.aten.sign.default),
    (torch.tanh, torch.ops.aten.tanh.default),
    (torch.sin, torch.ops.aten.sin.default),
    (torch.cos, torch.ops.aten.cos.default),
    (torch.sqrt, torch.ops.aten.sqrt.default),
    (
        torch.clone,
        torch.ops.aten.mul.Tensor,
    ),  # clone op can not be the output directly so expected is the op after it(aten.mul)
]


class TestUnaryOpsConverter(DispatchTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in unary_ops])
    def test_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return orig_op(x) * 2

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]
        _ = model(*inputs)
        self.run_test(model, inputs, expected_ops={expected_op})

    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in unary_ops])
    def test_dynamic_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return orig_op(x) * 2

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 8, 10],
            ],
            inputs_max=[
                [20, 12, 32],
            ],
            dtype_list=[
                torch.float16,
                torch.float16,
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(model, inputs_spec, expected_ops={expected_op})
