# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from .module import Module

# pylint: disable=C0103


class Identity(Module):
    """[summary]

    Parameters
    ----------
    Block : [type]
        [description]
    """

    def __init__(
        self,
        dtype="float16",
    ):
        """[summary]

        Parameters
        ----------
        in_channel : [type]
            [description]
        out_channel : [type]
            [description]
        bias : [type]
            [description]
        activation : [type]
            [description]
        weight_transpose : [type]
            [description]
        dtype : str, optional
            [description], by default "float16"

        Raises
        ------
        NotImplementedError
            [description]
        """
        super().__init__()

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 1
        data = args[0]
        return data
