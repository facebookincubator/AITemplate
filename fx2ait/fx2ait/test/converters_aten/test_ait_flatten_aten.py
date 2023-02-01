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
import torch.nn as nn
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestFlattenConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 1, 3),
            ("flatten_all", 0, 3),
        ]
    )
    def test_flatten(self, _, start_dim, end_dim):
        class TestModule(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        model = TestModule(start_dim, end_dim).cuda().half()
        inputs = (torch.randn(1, 2, 3, 1).half().cuda(),)

        self.run_test(model, inputs, expected_ops={torch.ops.aten.view.default})

    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 1, 3),
        ]
    )
    def test_flatten_with_dynamic_shape(self, _, start_dim, end_dim):
        class TestModule(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        model = TestModule(start_dim, end_dim).cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 2, 3, 4],
            ],
            inputs_max=[
                [10, 20, 3, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.view.default},
        )
