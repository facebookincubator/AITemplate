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
import unittest

import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase


class TestATenReshapeConverter(DispatchTestCase):
    def test_reshape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, (2, 12))

        size = (2, 3, 4)
        model = TestModule().cuda().half()
        inputs = (torch.randn(size).half().cuda(),)

        self.run_test(model, inputs, expected_ops={torch.ops.aten.view.default})

    def test_reshape_size(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                dim1_y = y.shape[0]
                return torch.reshape(x, (dim1_y, -1, 128))

        model = TestModule().cuda().half()
        inputs = (
            torch.randn(2, 10, 128).half().cuda(),
            torch.randn(2, 10, 128).half().cuda(),
        )

        self.run_test(model, inputs, expected_ops={torch.ops.aten.view.default})

    def test_reshape_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, (x.size(0), x.size(1) * x.size(2)))

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [10, 30, 4],
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

    def test_reshape_neg_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, (x.size(0), -1))

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [10, 30, 4],
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

    # TODO: trigger assertion in AIT: AssertionError: When there is no unknown index, we expect dim products to be equal, got current shape numel=2560 != new shape prod=256
    @unittest.skip
    def test_reshape_size_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                dim1_y = y.shape[0]
                return torch.reshape(x, (dim1_y, -1, 128))

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 10, 128],
                [2, 10, 128],
            ],
            inputs_max=[
                [20, 10, 128],
                [20, 10, 128],
            ],
            dtype_list=[
                torch.float16,
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.view.default},
        )
