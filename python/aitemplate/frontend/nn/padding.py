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
Padding related modules.
"""
from ...compiler.ops import ndhwc3to8, nhwc3to8
from .module import Module


class Nhwc3to8(Module):
    r"""Pads the input data with nhwc dimensions from 3 channels to 8 channels"""

    def __init__(self):
        super().__init__()
        self.op = nhwc3to8()

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)


class Ndhwc3to8(Module):
    r"""Pads the input data with ndhwc dimensions from 3 channels to 8 channels"""

    def __init__(self):
        super().__init__()
        self.op = ndhwc3to8()

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)
