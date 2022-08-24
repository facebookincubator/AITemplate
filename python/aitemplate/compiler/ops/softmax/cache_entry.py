# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from dataclasses import dataclass

# pylint: disable=C0103


@dataclass
class NormQueryEntry:
    """Query Entry

    Attributes
    ----------
    """

    dtype_in: int
    dtype_acc: int
    dtype_out: int
    rank: int
    op_type: str
    device: str
    exec_entry_sha1: str


@dataclass
class NormRecordEntry:
    """Record Entry

    Attributes
    ----------
    """

    exec_entry: str
    exec_entry_sha1: str
    dtype_in: int
    dtype_acc: int
    dtype_out: int
    rank: int
    op_type: str
    device: str
    algo: str
    workspace: int
