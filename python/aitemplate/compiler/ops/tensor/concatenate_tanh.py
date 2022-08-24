# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
summary
"""
from . import concatenate

# pylint: disable=C0103


class concatenate_tanh(concatenate):
    """_summary_

    Parameters
    ----------
    concatenate : _type_
        _description_
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "concatenate_tanh"
