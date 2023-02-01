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
from parameterized import param, parameterized


class TestClampConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0, use_clamp=True),
            param("min", min=0.5, use_clamp=False),
            param("max", max=0.5, use_clamp=True),
            param("minBiggerThanMax", min=1, max=0, use_clamp=False),
        ]
    )
    def test_clamp(self, name, min=None, max=None, use_clamp=True):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.clamp if use_clamp else torch.clip

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.op(x, min=min, max=max)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.clamp.default})

    @parameterized.expand(
        [
            param("default", min=-1, max=0, use_clamp=True),
            param("min", min=0.5, use_clamp=False),
            param("max", max=0.5, use_clamp=True),
            param("minBiggerThanMax", min=1, max=0, use_clamp=False),
        ]
    )
    def test_dynamic_clamp(self, name, min=None, max=None, use_clamp=True):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.clamp if use_clamp else torch.clip

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.op(x, min=min, max=max)

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

        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.clamp.default}
        )
