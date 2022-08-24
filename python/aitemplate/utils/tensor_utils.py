# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Util functions to handle tensor shapes.
"""


def wrap_dim(idx, rank):
    """
    Wrap tensor index, idx, if it's negative.
    """
    assert isinstance(idx, int)
    if idx < 0:
        idx = idx + rank
    assert idx < rank
    return idx
