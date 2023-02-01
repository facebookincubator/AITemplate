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
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestVarConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", dim=0, unbiased=False),
            param("unbiased", dim=0, unbiased=True),
            param("neg_dim", dim=-1, unbiased=True),
            param("keepdim", dim=0, unbiased=True, keepdim=True),
        ]
    )
    def test_var(self, name, dim, unbiased, keepdim=False):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.var(x, dim=dim, unbiased=unbiased, keepdim=keepdim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.var})

    @parameterized.expand(
        [
            param("default", dim=0, unbiased=False),
            param("unbiased", dim=0, unbiased=True),
            param("neg_dim", dim=-1, unbiased=True),
            param("keepdim", dim=0, unbiased=True, keepdim=True),
        ]
    )
    def test_var_call_method(self, name, dim, unbiased, keepdim=False):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.var})
