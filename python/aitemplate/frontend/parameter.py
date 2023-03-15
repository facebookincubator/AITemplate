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
Parameter definition.
"""
from aitemplate.compiler.base import Tensor


class Parameter:
    def __init__(self, shape, dtype, name=None, value=None):
        self._tensor = Tensor(shape=shape, dtype=dtype, name=name)
        self._value = value

    def tensor(self):
        return self._tensor

    def value(self):
        return self._value
