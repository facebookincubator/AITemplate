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
import torchvision
from fx2ait.example.benchmark_utils import benchmark_function, verify_accuracy


class TestResNet(unittest.TestCase):
    def test_resnet18(self):
        torch.manual_seed(0)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torchvision.models.resnet18()

            def forward(self, x):
                return self.mod(x)

        model = TestModule().cuda().half()
        inputs = [torch.randn(32, 3, 224, 224).half().cuda()]
        verify_accuracy(
            model,
            inputs,
            permute_inputs=[0, 2, 3, 1],
        )
        results = []
        for batch_size in [1, 8, 16, 32, 256, 512]:
            inputs = [torch.randn(batch_size, 3, 224, 224).half().cuda()]
            results.append(
                benchmark_function(
                    self.__class__.__name__,
                    100,
                    model,
                    inputs,
                    permute_inputs=[0, 2, 3, 1],
                )
            )
        for res in results:
            print(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
