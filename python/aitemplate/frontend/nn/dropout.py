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
"""Dropout/DropPath placeholder"""
from aitemplate.frontend.nn.module import Module

# pylint: disable=C0103


class Dropout(Module):
    r"""Dropout placeholder"""

    def __init__(
        self,
        p=0,
        dtype="float16",
    ):
        super().__init__()

    def forward(self, *args):
        r"""Not implemented."""
        assert len(args) == 1
        data = args[0]
        return data


class DropPath(Dropout):
    r"""DropPath placeholder"""

    def __init__(
        self,
        dtype="float16",
    ):
        super().__init__()
