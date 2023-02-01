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
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestLinalgConverter(AITTestCase):
    @parameterized.expand(
        [
            param(
                "l2_norm_dim_3",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=3,
                keepdims=False,
            ),
            param(
                "l2_norm_dim_2",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=2,
                keepdims=False,
            ),
            param(
                "l2_norm_dim_1",
                input_shape=[1, 100, 40, 40],
                ord=2,
                dim=1,
                keepdims=True,
            ),
        ]
    )
    def test_linalg_norm(
        self, test_name, input_shape, ord=None, dim=None, keepdims=False
    ):
        class TestModule(torch.nn.Module):
            def __init__(self, ord, dim, keepdims):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.linalg.norm(x, ord, dim, keepdims)

        model = TestModule(ord, dim, keepdims).cuda().half()
        inputs = [
            torch.randn(input_shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.linalg_norm})
