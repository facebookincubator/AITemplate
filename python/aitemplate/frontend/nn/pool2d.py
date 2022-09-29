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
"""
pool2d-family modules.
"""
from ...compiler.ops import avg_pool2d, max_pool2d
from .module import Module


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.op = max_pool2d(kernel_size, stride, padding)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.op = avg_pool2d(kernel_size, stride, padding)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)
