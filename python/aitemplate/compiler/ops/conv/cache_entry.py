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
Cache entry for conv2d.
"""
from dataclasses import dataclass

# pylint: disable=C0103


@dataclass
class ConvQueryEntry:
    """Query Entry"""

    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    kh: int
    kw: int
    co: int
    strideh: int
    stridew: int
    padh: int
    padw: int
    dilateh: int
    dilatew: int
    op_type: str
    device: str
    epilogue: int
    split_k: int
    exec_entry_sha1: str


@dataclass
class ConvRecordEntry:
    """Record Entry"""

    exec_entry: str
    exec_entry_sha1: str
    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    kh: int
    kw: int
    co: int
    strideh: int
    stridew: int
    padh: int
    padw: int
    dilateh: int
    dilatew: int
    op_type: str
    epilogue: int
    device: str
    algo: str
    workspace: int
    split_k: int


@dataclass
class Conv3dQueryEntry:
    """Query Entry"""

    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    kd: int
    kh: int
    kw: int
    co: int
    stride_d: int
    stride_h: int
    stride_w: int
    pad_d: int
    pad_h: int
    pad_w: int
    dilate_d: int
    dilate_h: int
    dilate_w: int
    op_type: str
    device: str
    epilogue: int
    split_k: int
    exec_entry_sha1: str


@dataclass
class Conv3dRecordEntry:
    """Record Entry"""

    exec_entry: str
    exec_entry_sha1: str
    dtype_a: int
    dtype_b: int
    dtype_c: int
    dtype_acc: int
    major_a: int
    major_b: int
    major_c: int
    kd: int
    kh: int
    kw: int
    co: int
    stride_d: int
    stride_h: int
    stride_w: int
    pad_d: int
    pad_h: int
    pad_w: int
    dilate_d: int
    dilate_h: int
    dilate_w: int
    op_type: str
    epilogue: int
    device: str
    algo: str
    workspace: int
    split_k: int
