# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from dataclasses import dataclass

# pylint: disable=C0103


@dataclass
class ConvQueryEntry:
    """Query Entry

    Attributes
    ----------
    """

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
    stride: int
    pad: int
    dilate: int
    op_type: str
    device: str
    epilogue: int
    split_k: int
    exec_entry_sha1: str


@dataclass
class ConvRecordEntry:
    """Record Entry

    Attributes
    ----------
    """

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
    stride: int
    pad: int
    dilate: int
    op_type: str
    epilogue: int
    device: str
    algo: str
    workspace: int
    split_k: int
