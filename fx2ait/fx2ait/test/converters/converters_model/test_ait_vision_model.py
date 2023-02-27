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
import torchvision
from fx2ait.tools.common_fx2ait import AITTestCase


class TestVisionModelConverter(AITTestCase):
    def test_resnet50(self):
        torch.manual_seed(0)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torchvision.models.resnet18()

            def forward(self, x):
                return self.mod(x)

        model = TestModule().cuda().half()
        inputs = [torch.randn(32, 3, 224, 224).half().cuda()]
        self.run_test(
            model,
            inputs,
            expected_ops={},
            permute_outputs=None,
        )
