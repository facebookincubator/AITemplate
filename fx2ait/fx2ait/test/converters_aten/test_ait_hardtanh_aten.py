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


class TestHardTanhConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", min=-1.5, max=3),
            param("min", min=-1.5),
            param("max", max=3),
        ]
    )
    def test_hardtanh(self, name, min=-1, max=1):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.nn.Hardtanh(min, max)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.op(x)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={torch.ops.aten.hardtanh.default})

    @parameterized.expand(
        [
            param("default", min=-1.2, max=2),
            param("min", min=-1.2),
            param("max", max=2),
        ]
    )
    def test_dynamic_hardtanh(self, name, min=-1, max=1):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.nn.Hardtanh(min, max)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.op(x)

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
            model, inputs_spec, expected_ops={torch.ops.aten.hardtanh.default}
        )
