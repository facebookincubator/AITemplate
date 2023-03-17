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
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestSoftmaxConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", dim=1),
            param("neg", dim=-1),
        ]
    )
    def test_softmax(self, name, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=dim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.softmax})

    @parameterized.expand(
        [
            param("default", dim=2),
            param("neg", dim=-3),
        ]
    )
    def test_softmax_not_last_dim(self, name, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=dim)

        model = TestModule().cuda().half()

        # Test static use case
        inputs = [
            torch.randn(2, 3, 5, 1, 1).half().cuda(),
        ]
        self.run_test(model, inputs, expected_ops={acc_ops.softmax})

        # Test dynamic use case
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 5, 1, 1],
            ],
            inputs_max=[
                [20, 10, 5, 1, 1],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={acc_ops.softmax},
        )

    @parameterized.expand(
        [
            param("default", dim=2),
            param("neg", dim=-3),
        ]
    )
    def test_softmax_expected_failure(self, name, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=dim)

        model = TestModule().cuda().half()

        inputs = [
            torch.randn(2, 3, 5, 2, 1).half().cuda(),
        ]
        with self.assertRaises(ValueError):
            self.run_test(model, inputs, expected_ops={acc_ops.softmax})

    @parameterized.expand(
        [
            param("default", dim=2),
            param("neg", dim=-3),
        ]
    )
    def test_softmax_expected_failure_dynamic(self, name, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=dim)

        model = TestModule().cuda().half()

        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 5, 2, 1],
            ],
            inputs_max=[
                [20, 10, 5, 4, 1],
            ],
            dtype_list=[
                torch.float16,
            ],
        )
        with self.assertRaises(ValueError):
            self.run_test_with_dynamic_shape(
                model,
                inputs_spec,
                expected_ops={acc_ops.softmax},
            )
