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
Util functions to handle tensor shapes.
"""


def wrap_dim(idx, rank):
    """
    Wrap tensor index, idx, if it's negative.
    """
    assert isinstance(idx, int), "idx must be int, but got {}".format(type(idx))
    if idx < 0:
        idx = idx + rank
    assert idx < rank, "idx {} out of range; rank {}".format(idx, rank)
    return idx
