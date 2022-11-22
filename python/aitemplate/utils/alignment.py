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
Util functions to handle alignment.
"""
# Currently read4, add2 is best for both backend, so two backend seems identical.
# They may diverge when we got deeper understanding / further optimization.
ALIGNMENTS = [
    8,
    4,
    2,
    1,
]


def find_max_alignment(number: int) -> int:
    """
    Return the first alignment value that meets the alignment requirement
    for accessing the `number` of elements. This is dtype dependent.
    """
    for alignment in ALIGNMENTS:
        if number % alignment == 0:
            return alignment
    return 1
