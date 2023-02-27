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
from typing import List

import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestTopkConverter(AITTestCase):
    @parameterized.expand(
        [
            [[4], 1],
            [[6], 3],
            [[6], 6],
        ]
    )
    def test_simple(self, input: List[int], k: int) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, indices = torch.topk(x, k)
                return indices

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.topk})

    @parameterized.expand(
        [
            [[2, 4], 1],
            [[2, 4], 2],
            [[3, 3], 3],
        ]
    )
    def test_multi_dimensional(self, input: List[int], k: int) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, indices = torch.topk(x, k)
                return indices

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.topk})

    ##TODO results mismatch.(P537992074)
    # def test_multi_dimensional_dynamic_shape(self) -> None:
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             values, indices = torch.topk(x, 1)
    #             return indices

    #     model = TestModule().cuda()
    #     inputs = [
    #         [
    #             torch.randn((2, 4)).half().cuda(),
    #         ],
    #         [
    #             torch.randn((20, 4)).half().cuda(),
    #         ],
    #     ]
    #     self.run_test_with_dynamic_shape(model, inputs, expected_ops={acc_ops.topk})
