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
Frontend for attention module
"""
from aitemplate.compiler import ops
from aitemplate.compiler.ops import flash_attention
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.dropout import Dropout
from aitemplate.frontend.nn.linear import Linear
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter
from aitemplate.testing import detect_target


class FlashAttention(Module):
    r"""FlashAttention provides an implementation for fused
    multi-head attention module:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK}{\sqrt(d)}) * V

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    """

    def __init__(
        self,
        batch_size,
        max_seq_len,
        dropout=0,
        causal=False,
        dtype="float16",
    ):
        """Initialize attention module, create a tensor for seqlen"""
        super().__init__()
        self.cu_length = Parameter(shape=[batch_size + 1], dtype="int32")
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
        return self.op(x, self.cu_length.tensor())


class MultiheadAttention(Module):
    r"""Multi-Head Attention.

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

    USE_CUDA = None

    def __init__(
        self,
        dim,
        batch_size,
        seq_len,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        has_residual=True,
        causal=False,
        mask_seq=0,
        use_mem_eff=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divisible by num_heads {num_heads}"
        if MultiheadAttention.USE_CUDA is None:
            MultiheadAttention.USE_CUDA = detect_target().name() == "cuda"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.causal = causal
        self.has_residual = has_residual
        self.mask_seq = mask_seq
        self.use_mem_eff = use_mem_eff

        flash_head_dims = {8, 16, 32, 64, 128}
        # simple heuristic, may need refinement
        self.use_flash = (
            not (seq_len >= 512 and batch_size <= 2)
        ) and head_dim in flash_head_dims
        # odd seq try use flash
        if seq_len % 2 == 1:
            self.use_flash = True

        if use_mem_eff:
            self.op = ops.mem_eff_attention(
                causal=causal,
            )
            self.use_flash = False
        else:
            self.op = flash_attention(
                batch_size=batch_size,
                dropout=attn_drop,
                max_seq_len=seq_len,
                causal=causal,
            )
        # cu_length: the cumulative sequence lengths, used to index into hidden_states.
        self.cu_length = Parameter(shape=[batch_size + 1], dtype="int32")
        if self.mask_seq:
            self.output_mask = Parameter(
                shape=[mask_seq, num_heads, head_dim], dtype="float16"
            )

        if self.USE_CUDA:
            # on CUDA flash_attention needs packed QKV as input,
            # then do split + permute inside flash_attn
            # input: (B, S, H)
            # output: (B*S, 3, num_heads, head_dim)
            if self.use_flash:
                self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
            else:
                self.qkv = Linear(
                    dim,
                    dim * 3,
                    specialization="permute",
                    shape=(seq_len, 3, self.num_heads),
                )
        else:
            # on ROCM ck attention (bmm_softmax_bmm) takes three inputs (Q, K, V)
            # here we generate packed QKV for splitting
            # input: (B, seqlen, dim) -> (B*seqlen, dim)
            # gemm: (B*seqlen, 3*dim)
            # reshape to: (B, seqlen, 3, num_heads, head_dim)
            # output: (3, B, num_heads, seqlen, head_dim)
            self.qkv = Linear(
                dim,
                dim * 3,
                specialization="permute",
                shape=(seq_len, 3, self.num_heads),
                layout="m2n3",
            )

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, specialization="add" if has_residual else None)
        self.proj_drop = Dropout(proj_drop)

    def get_shape(self, x):
        shape = [it.value() for it in x._attrs["shape"]]
        return shape

    def qkv_proj(self, x):
        if self.USE_CUDA:
            if self.use_flash:
                batch, seq, hidden = self.get_shape(x)
                out = self.qkv(x)
                return ops.reshape()(
                    out, [int(batch * seq), 3, self.num_heads, hidden // self.num_heads]
                )
            else:
                batch, seq, hidden = self.get_shape(x)
                x = ops.reshape()(x, [-1, hidden])
                return self.qkv(x)
        else:
            return self.qkv(x)

    def attention(self, x):
        # fused attention
        # output: (B, Seqlen, num_heads, head_dim)
        if self.USE_CUDA and self.use_flash:
            # input(x): (B*seqlen, 3, num_heads, head_dim)
            # output: (B, Seqlen, num_heads, head_dim)
            return self.op(x, self.cu_length.tensor())
        elif self.USE_CUDA and self.use_mem_eff:
            (q, k, v) = ops.split()(x, 1, dim=0)
            _, b, num_heads, seqlen, d = self.get_shape(q)
            return self.op(
                ops.reshape()(q, [b, -1, seqlen, d]),
                ops.reshape()(k, [b, -1, seqlen, d]),
                ops.reshape()(v, [b, -1, seqlen, d]),
            )
        else:
            # input(q/k/v): (B*num_heads, seqlen, head_dim)
            # attn = (B, S, H) * (B, S, H) = (B, S, S) #RCR
            # softmax on dim -1 (B, S, S)
            # attn@v: (B, S, S) * (B, S, H) = (B, S, H) #RRR
            # reshape: (B, num_head, seqlen, head_dim)
            # permute: (B, Seqlen, num_heads, head_dim)
            if self.USE_CUDA:
                scale = Tensor(
                    shape=[], dtype="float16", name="scale", value=self.scale
                )
                # [3, b, num_heads, seqlen, d]
                _, b, num_heads, seqlen, d = self.get_shape(x)
                # [3 * b * num_heads, seqlen, d]
                x = ops.reshape()(x, [-1, seqlen, d])
                (q, k, v) = ops.split()(x, b * num_heads, dim=0)
                qk = ops.bmm_rcr()(q, k)
                score = ops.elementwise(FuncEnum.MUL)(qk, scale)
                score = ops.softmax()(score, -1)
                out = ops.bmm_rrr_permute((num_heads,))(score, v)
            else:
                (q, k, v) = ops.split()(x, 1, dim=0)
                _, _, _, seqlen, d = self.get_shape(q)
                OP = ops.bmm_softmax_bmm_permute(
                    shape=(self.num_heads,),
                    scale=self.scale,
                    causal=self.causal,
                )
                out = OP(
                    ops.reshape()(q, [-1, seqlen, d]),
                    ops.reshape()(k, [-1, seqlen, d]),
                    ops.reshape()(v, [-1, seqlen, d]),
                )
            return out

    def forward(self, *args):
        """forward pass for calling mha module"""
        assert len(args) >= 1
        x = args[0]
        batch, seq, hidden = self.get_shape(x)
        qkv = self.qkv_proj(x)
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


class CrossAttention(Module):
    r"""Cross Multi-head Attention.

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

        self.op = ops.mem_eff_attention(causal=causal)

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

        query = ops.permute()(
            ops.reshape()(query, [-1, seqlen, self.num_heads, head_dim]), [0, 2, 1, 3]
        )
        key = ops.permute()(
            ops.reshape()(key, [-1, seqlen_kv, self.num_heads, head_dim]), [0, 2, 1, 3]
        )
        value = ops.permute()(
            ops.reshape()(value, [-1, seqlen_kv, self.num_heads, head_dim]),
            [0, 2, 1, 3],
        )
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


class ScaledDotProductAttention(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k, v):
        attn = ops.mem_eff_attention(causal=False)(q, k, v)
        return attn
