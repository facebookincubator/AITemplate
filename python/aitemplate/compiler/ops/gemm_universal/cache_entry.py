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
GEMM profiling cache entries
"""
from dataclasses import dataclass


@dataclass
class GemmQueryEntry:
    """GEMM query entry"""

    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    op_type: str
    device: str
    epilogue: int
    exec_entry_sha1: str
    pshape: str


@dataclass
class GemmRecordEntry:
    """Profile result record entry"""

    exec_entry: str
    exec_entry_sha1: str
    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    op_type: str
    epilogue: int
    pshape: str
    device: str
    algo: str
    workspace: int
    split_k: int
