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
from typing import List, Union

import torch
from fx2ait.acc_tracer import ait_acc_ops

from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_fx2ait import AITTestCase

from parameterized import parameterized


class TestSplitConverter(AITTestCase):
    @parameterized.expand(
        [
            [[2, 10], [2, 3, 5]],
            [[2, 10], 2],
            [[2, 10], 3],
        ]
    )
    def test_with_dim(
        self, input_shape: List[int], split_size_or_sections: Union[int, List[int]]
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, split_size_or_sections, dim=1)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={ait_acc_ops.split})

    @parameterized.expand(
        [
            [[10], [2, 3, 5]],
            [[10], 2],
            [[10], 3],
        ]
    )
    def test_without_dim(
        self, input_shape: List[int], split_size_or_sections: Union[int, List[int]]
    ) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, split_size_or_sections)

        model = TestModule().cuda()
        inputs = [
            torch.randn(*input_shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={ait_acc_ops.split})

    def test_with_dim_dynamic_shape(self) -> None:
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.split(x, 2, dim=1)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 10],
            ],
            inputs_max=[
                [20, 10],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={ait_acc_ops.split}
        )
