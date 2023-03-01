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
Frontend for vanilla attention module
"""
from functools import partial

from ...compiler import ops
from .. import Tensor
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .parameter import Parameter

# pylint: disable=C0103


def _get_dim(it):
    try:
        return it.value()
    except AttributeError:
        return -1


def _get_shape(x):
    shape = [_get_dim(it) for it in x._attrs["shape"]]
    return shape


def vanilla_attention(
    q: Tensor, k: Tensor, v: Tensor, scale: float = None, attn_mask: Tensor = None
) -> Tensor:
    """Vanilla attention in the most basic form.
    q,k,v: batch, seqlen, num_heads, head_dim
        Either batch or sequence dimension could be variable (but not both)
    attn_mask: attention mask is *added* to the attention,
        use 0 and -inf to mask a sequence index
    """
    batch_name, seq_name = [it._attrs["name"] for it in q._attrs["shape"]][0:2]
    B, N, G, D = _get_shape(q)
    B, M, _, _ = _get_shape(k)
    BG = B * G
    if BG < 0:
        BG = -1
    C = G * D
    if scale is None:
        scale = D ** (-0.5)
    q = q * scale

    q = ops.permute()(q, [0, 2, 1, 3])  # BxGxNxD
    q = ops.reshape()(q, (BG, N, D))  # BGxNxD

    k = ops.reshape()(k, (B, M, C))  # BxMxGD
    k = ops.permute021()(k)  # BxGDxM
    k = ops.reshape()(k, (BG, D, M))  # BGxDxM

    attention = ops.bmm_rrr()(q, k)  # BGxNxM
    if attn_mask is not None:
        attention = attention + attn_mask
    attention = ops.softmax()(attention, -1)  # BGxNxM

    v = ops.reshape()(v, (B, M, C))  # BxMxGD
    v = ops.permute021()(v)  # BxGDxM
    v = ops.reshape()(v, (BG, D, M))  # BGxDxM

    out = ops.bmm_rcr()(v, attention)  # BGxDxN
    out = ops.reshape()(out, (B, C, N))  # BxGDxN == BxCxN
    out = ops.permute021()(out)  # BxNxC
    out._attrs["shape"][0]._attrs["name"] = batch_name
    out._attrs["shape"][1]._attrs["name"] = seq_name
    return out


class VanillaMultiheadAttention(Module):
    r"""Vanilla Multi-Head Attention.

    Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        dim: total dimension of the model
        batch_size: batch size
        seq_len: sequence length
        num_heads: Number of parallel attention heads. Default: 8
        qkv_bias: whether to add bias to QKV. Default: False
        attn_drop: Dropout probability on attention output weights. Default: ``0.0`` (no dropout).
        proj_drop: Dropout probability on projection layers. Default: ``0.0`` (no dropout).
        has_residual: has or has no residual. Default: `True`.
        causal: default: `False`.
        attn_mask: Attention mask. If causal this should be a tensor of shape [1, seq_len, seq_len] filled with -inf and 0
        mask_seq: sequence mask, default: ``0``.
    """

    def __init__(
        self,
        dim,
        batch_size=-1,
        seq_len=-1,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        has_residual=True,
        causal=False,
        attn_mask: Tensor = None,
        mask_seq=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.causal = causal
        self.has_residual = has_residual
        self.mask_seq = mask_seq

        if causal:
            assert (
                attn_mask is not None
            ), f"Missing attn_mask=Tensor(shape=1x{seq_len}x{seq_len})"
            self.op = partial(vanilla_attention, attn_mask=attn_mask)
        else:
            self.op = vanilla_attention

        if self.mask_seq:
            self.output_mask = Parameter(
                shape=[mask_seq, num_heads, head_dim], dtype="float16"
            )
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, specialization="add" if has_residual else None)
        self.proj_drop = Dropout(proj_drop)

    def get_shape(self, x):
        return _get_shape(x)

    def attention(self, x):
        b, seqlen, d = self.get_shape(x)
        hidden = d // 3
        x = ops.reshape()(x, [-1, 3, hidden])
        (q, k, v) = ops.split()(x, 1, dim=1)
        return self.op(
            ops.reshape()(q, [b, seqlen, self.num_heads, hidden // self.num_heads]),
            ops.reshape()(k, [b, seqlen, self.num_heads, hidden // self.num_heads]),
            ops.reshape()(v, [b, seqlen, self.num_heads, hidden // self.num_heads]),
            self.scale,
        )

    def forward(self, *args):
        """forward pass for calling mha module"""
        assert len(args) >= 1
        x = args[0]
        batch, seq, hidden = self.get_shape(x)
        qkv = self.qkv(x)
        if self.mask_seq:
            total = self.get_shape(qkv)[0]
            qkv = ops.dynamic_slice()(
                qkv,
                start_indices=[0, 0, 0, 0],
                end_indices=[total - self.mask_seq, None, None, None],
            )
        attn_output = self.attention(qkv)
        if self.mask_seq:
            attn_output = ops.concatenate()(
                [attn_output, self.output_mask.tensor()], dim=0
            )
        attn_output = ops.reshape()(attn_output, [batch * seq, -1])
        if self.has_residual:
            assert len(args) == 2
            x = self.proj(attn_output, args[1])
        else:
            x = self.proj(attn_output)
        x = self.proj_drop(x)
        x = ops.reshape()(x, [batch, seq, hidden])
        return x


class VanillaCrossAttention(Module):
    r"""Vanilla Cross Multi-head Attention.

    Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        dim: total dimension of the model
        batch_size: batch size
        seq_len: sequence length
        num_heads: Number of parallel attention heads. Default: 8
        qkv_bias: whether to add bias to QKV. Default: False
        attn_drop: Dropout probability on attention output weights. Default: ``0.0`` (no dropout).
        proj_drop: Dropout probability on projection layers. Default: ``0.0`` (no dropout).
        has_residual: has or has no residual. Default: `True`.
        causal: default: `False`.
        mask_seq: sequence mask, default: ``0``.
    """

    def __init__(
        self,
        dim,
        seq_len,
        seq_len_kv,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        has_residual=True,
        causal=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.causal = causal
        self.has_residual = has_residual
        self.dim = dim
        self.seqlen = seq_len
        self.seqlen_kv = seq_len_kv

        assert not causal, "Causal not implemented"
        self.op = vanilla_attention

        self.proj_q = Linear(
            dim,
            dim,
            bias=qkv_bias,
        )
        self.proj_k = Linear(
            dim,
            dim,
            bias=qkv_bias,
        )
        self.proj_v = Linear(
            dim,
            dim,
            bias=qkv_bias,
        )

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, specialization="add" if has_residual else None)
        self.proj_drop = Dropout(proj_drop)

    def attention(self, q, k, v):
        seqlen = self.seqlen
        seqlen_kv = self.seqlen_kv
        head_dim = self.dim // self.num_heads

        query = self.proj_q(q)
        key = self.proj_k(k)
        value = self.proj_v(v)

        query = ops.reshape()(query, [-1, seqlen, self.num_heads, head_dim])
        key = ops.reshape()(key, [-1, seqlen_kv, self.num_heads, head_dim])
        value = ops.reshape()(value, [-1, seqlen_kv, self.num_heads, head_dim])
        return self.op(query, key, value)

    def forward(self, *args):
        """forward pass for calling mha module"""
        assert len(args) >= 3
        x = args[0]
        seq = self.seqlen
        attn_output = self.attention(args[0], args[1], args[2])
        attn_output = ops.reshape()(attn_output, [-1, seq, self.dim])

        if self.has_residual:
            assert len(args) == 4
            x = self.proj(attn_output, args[3])
        else:
            x = self.proj(attn_output)
        x = self.proj_drop(x)
        x = ops.reshape()(x, [-1, seq, self.dim])
        return x
