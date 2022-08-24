# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Frontend for attention module
"""
from ...compiler.ops import flash_attention
from .module import Module
from .parameter import Parameter

# pylint: disable=C0103


class FlashAttention(Module):
    """[summary]

    Parameters
    ----------
    Block : [type]
        [description]
    """

    def __init__(
        self,
        batch_size,
        max_seq_len,
        dropout=0,
        causal=False,
        dtype="float16",
    ):
        """Initilize attention module, create a tensor for seqlen"""
        super().__init__()
        self.seq_len = Parameter(shape=[batch_size + 1], dtype="int32")
        self.op = flash_attention(
            batch_size=batch_size,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=causal,
        )

    def forward(self, *args):
        """forward pass for calling attention op"""
        assert len(args) == 1
        x = args[0]
        return self.op(x, self.seq_len.tensor())
