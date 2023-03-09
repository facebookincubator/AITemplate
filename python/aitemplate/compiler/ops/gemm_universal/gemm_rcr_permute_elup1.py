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
A specialization of gemm_rcr_permute applying ELU + 1 as epilogue.
"""

from aitemplate.compiler.ops.gemm_universal import gemm_rcr_permute

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_permute_elup1(gemm_rcr_permute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attrs["op"] = "gemm_rcr_permute_elup1"
        self._attrs["epilogue"] = "LinearCombinationELUp1"
