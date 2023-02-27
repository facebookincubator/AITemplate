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
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestPermuteConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param((128, 512), (1, 0)),
            param((80, 300, 2), (0, 2, 1)),
            param((80, 300, 2), (1, 0, 2)),
            param((80, 300, 2), (2, 1, 0)),
            param((5, 113, 15, 31), (0, 2, 1, 3)),
            param((2, 3, 4, 5), (3, 2, 1, 0)),
            param((3, 5, 128, 514), (2, 3, 0, 1)),
            param((32, 12, 4096, 64), (0, 2, 1, 3)),
            param((3, 1, 113, 15, 64), (2, 0, 3, 1, 4)),
        ]
    )
    def test_permute(self, shape, dims):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.permute(x, dims)

        model = TestModule().cuda()
        inputs = [
            torch.randn(shape).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.permute.default})

    @parameterized.expand(
        [
            param((128, 500), (256, 512), (1, 0)),
            param((80, 300, 2), (98, 512, 20), (0, 2, 1)),
            param((80, 300, 2), (98, 512, 20), (1, 0, 2)),
            param((80, 300, 2), (98, 512, 20), (2, 1, 0)),
            param((3, 5, 128, 512), (6, 10, 256, 520), (0, 2, 1, 3)),
            param((3, 5, 128, 512), (6, 10, 256, 520), (3, 2, 1, 0)),
            param((3, 5, 128, 512), (6, 10, 256, 520), (2, 3, 0, 1)),
            param((3, 1, 113, 15, 64), (6, 10, 128, 16, 128), (2, 0, 3, 1, 4)),
        ]
    )
    def test_permute_dynamic_shape(self, input_min, input_max, dims):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.permute(x, dims)

        model = TestModule().cuda()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                input_min,
            ],
            inputs_max=[
                input_max,
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model, inputs_spec, expected_ops={torch.ops.aten.permute.default}
        )
