# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from dataclasses import dataclass


@dataclass
class GemmQueryEntry:
    """_summary_

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
    op_type: str
    device: str
    epilogue: int
    exec_entry_sha1: str


@dataclass
class GemmRecordEntry:
    """_summary_

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
    op_type: str
    epilogue: int
    device: str
    algo: str
    workspace: int
    split_k: int
