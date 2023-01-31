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
from torch import nn


class TestSigmoidConverter(AITTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        model = Sigmoid().cuda()
        inputs = [torch.randn(1, 2, 3).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.sigmoid})
