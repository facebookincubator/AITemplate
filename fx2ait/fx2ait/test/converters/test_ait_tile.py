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
from parameterized import parameterized
from torch import nn


class TestTile(AITTestCase):
    @parameterized.expand(
        [
            ("same_num_dims", (2, 2, 3), (1, 2, 2)),
            (
                "less_dims",
                (2, 2, 3),
                (
                    1,
                    2,
                ),
            ),
            ("more_dims", (2, 3), (1, 2, 2, 1)),
        ]
    )
    def test_tile(self, _, input_shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                x = x + x  # avoid input shape infer error from AIT
                return torch.tile(x, self.dims)

        model = Tile(dims).half().cuda()
        inputs = [torch.randn(*input_shape).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.add, acc_ops.tile})
