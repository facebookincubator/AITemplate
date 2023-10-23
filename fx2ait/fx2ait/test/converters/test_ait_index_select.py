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


class TestIndexSelectConverter(AITTestCase):
    @parameterized.expand(
        [
            param(
                "first_dim",
                torch.randn(5, 10, 20),
                0,
                torch.randint(low=0, high=5, size=(3,)),
            ),
            param(
                "mid_dim",
                torch.randn(5, 10, 20),
                1,
                torch.randint(low=0, high=10, size=(20,)),
            ),
            param(
                "last_dim",
                torch.randn(5, 10, 20),
                2,
                torch.randint(low=0, high=20, size=(10,)),
            ),
        ]
    )
    def test_index_select(self, _, inp, dim, index):
        class TestModule(torch.nn.Module):
            def forward(
                self,
                inp: torch.Tensor,
                index: torch.Tensor,
            ) -> torch.Tensor:
                return torch.index_select(inp, dim, index=index)

        model = TestModule().eval().half().cuda()
        inputs = [inp.cuda(), index.cuda()]

        self.run_test(model, inputs, expected_ops={acc_ops.index_select})
