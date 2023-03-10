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

from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestReshapeConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 3, 4], [6, 4]],
            [[2, 3, 4], [2, 12]],
            [[2, 3, 4], [24]],
            [[2, 3, 4], [-1, 4]],
            [[2, 3, 4], [2, -1]],
            [[2, 3, 4], [-1]],
        ]
    )
    def test_simple(self, original_shape: List[int], final_shape: List[int]) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, final_shape)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*original_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.reshape})

    def test_with_getitem_size(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                d0 = y.size(dim=0)
                d1 = y.size(dim=1)
                return x.reshape(d0, d1)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
            torch.randn(6, 4).half().cuda(),
        ]
        self.run_test(
            model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
        )

    def test_with_getitem_reshape_dim0(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * d2
                return x.reshape(d0, d)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 3, 4).half().cuda(),
        ]
        self.run_test(
            model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
        )

    def test_reshape_with_non_int_param(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=1)
                return x.reshape(d0 * 8)

        model = TestModule().cuda()
        inputs = [
            torch.randn(2, 4, 4).half().cuda(),
            torch.randn(4, 8).half().cuda(),
        ]
        self.run_test(
            model, inputs, expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem}
        )

    def test_with_getitem_reshape_dim0_dynamic(self) -> None:
        class TestSimpleModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * d2
                return x.reshape(d0, d)

        model = TestSimpleModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 3, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem},
        )

        class TestComplexModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * (d2 + d1 - 3)  # d2+d1-3=d2
                return x.reshape(d0, d)

        model = TestComplexModule().cuda()
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem},
        )

    def test_with_getitem_reshape_dim01_dynamic(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * d2
                return x.reshape(d0, d)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 30, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem},
        )

        class TestComplexModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d0 = x.size(dim=0)
                d1 = x.size(dim=1)
                d2 = x.size(dim=2)
                d = d1 * (d2 - d0 + d0)
                return x.reshape(d0, d)

        model = TestComplexModule().cuda()
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={acc_ops.reshape, acc_ops.size, acc_ops.getitem},
        )
