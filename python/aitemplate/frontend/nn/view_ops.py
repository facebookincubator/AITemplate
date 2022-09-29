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
View-related modules.
"""
from ...compiler.ops import flatten, reshape
from .module import Module


class Reshape(Module):
    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class View(Module):
    def __init__(self):
        super().__init__()
        self.op = reshape()

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        shape = args[1]
        return self.op(x, shape)


class Flatten(Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.op = flatten(start_dim, end_dim)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)
