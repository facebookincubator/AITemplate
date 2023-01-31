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
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestSumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
            ["keepdim_2d", (0, 1), True],
            ["nokeepdim_2d", (0, 1), False],
            ["negative_2d", (-1, -2), False],
            ["keepdim_3d", (0, 1, 2), True],
        ]
    )
    def test_sum(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sum(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.sum.dim_IntList})

    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
            ["keepdim_2d", (0, 1), True],
            ["nokeepdim_2d", (0, 1), False],
            ["negative_2d", (-1, -2), False],
            ["keepdim_3d", (0, 1, 2), True],
        ]
    )
    def test_dynamic_sum(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sum(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        # The last dim has to be static to pre-compute vector_length:
        # https://fburl.com/code/1x07doen
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 6, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.sum.dim_IntList}
        )


class TestMeanConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
            ["keepdim_2d", (0, 1), True],
            ["nokeepdim_2d", (0, 1), False],
            ["negative_2d", (-1, -2), False],
            ["keepdim_3d", (0, 1, 2), True],
        ]
    )
    def test_mean(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mean(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.mean.dim})

    @parameterized.expand(
        [
            ["default", (1), False],
            ["keepdim", (1), True],
            ["negative_dim", (-1), False],
            ["keepdim_2d", (0, 1), True],
            ["nokeepdim_2d", (0, 1), False],
            ["negative_2d", (-1, -2), False],
            ["keepdim_3d", (0, 1, 2), True],
        ]
    )
    def test_dynamic_mean(self, test_name, dim, keepdim):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mean(x, dim=dim, keepdim=keepdim)

        model = TestModule().cuda()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [20, 6, 8],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.mean.dim}
        )
