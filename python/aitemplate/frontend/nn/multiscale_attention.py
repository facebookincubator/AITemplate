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
Frontend for multi-scale attention module
AIT implementation for MViT:
https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/vision_transformers.py
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.public import permute
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.activation import GELU
from aitemplate.frontend.nn.batch_norm import BatchNorm1d, BatchNorm3d
from aitemplate.frontend.nn.conv3d import Conv3d
from aitemplate.frontend.nn.dropout import Dropout, DropPath
from aitemplate.frontend.nn.identity import Identity
from aitemplate.frontend.nn.layer_norm import LayerNorm
from aitemplate.frontend.nn.linear import Linear
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.pool3d import MaxPool3d

_LOGGER = logging.getLogger(__name__)


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def ait_ncl2nlc(x):
    return permute()(x, [0, 2, 1])


def _unsqueeze_dims(x):
    tensor_dim = len(get_shape(x))
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        x = ops.unsqueeze(dim=1)(x)
    else:
        raise NotImplementedError(f"Unsupported input dimension {get_shape(x)}")
    return x, tensor_dim


def _squeeze_dims(x, tensor_dim):
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        x = ops.squeeze(dim=1)(x)
    else:
        raise NotImplementedError(f"Unsupported input dimension {get_shape(x)}")
    return x


class Mlp(Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.

    ::

                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Module = GELU,
        dropout_rate: float = 0.0,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # TODO fc1 bias is set to zeros; unset if bias_on is True

        self.fc1 = Linear(
            in_features,
            hidden_features,
            bias=bias_on,
        )
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, bias=bias_on)

        if self.dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)

        assert self.dropout_rate == 0.0

        if self.dropout_rate > 0.0:
            x = self.dropout(x)

        x = self.fc2(x)

        if self.dropout_rate > 0.0:
            x = self.dropout(x)

        return x


class _AttentionPool(Module):
    def __init__(
        self,
        pool: Optional[Module],
        has_cls_embed: bool,
        norm: Optional[Module],
    ) -> None:
        """Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


        Params:
            pool (Optional[Callable]): Pool operation that is applied to the input tensor.
                If pool is none, return the input tensor.
            has_cls_embed (bool): Whether the input tensor contains cls token. Pool
                operation excludes cls token.
            norm: (Optional[Callable]): Optional normalization operation applied to
            tensor after pool.
        """
        super().__init__()
        self.has_pool = pool is not None
        self.pool = pool if pool is not None else Identity()

        self.has_cls_embed = has_cls_embed
        if norm is not None:
            self.norm_before_pool = isinstance(norm, (BatchNorm3d, Identity))
            self.has_norm = True
            self.norm = norm
        else:
            self.norm_before_pool = False
            self.has_norm = False
            self.norm = Identity

    def forward(self, tensor: Tensor, thw_shape: List[int]) -> Tuple[Tensor, List[int]]:
        """
        Args:
            tensor (Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).

        Returns:
            tensor (Tensor): Input tensor after pool.
            thw_shape (List[int]): Output tensor shape (before flattening).
        """
        if not self.has_pool:
            return tensor, thw_shape

        tensor, tensor_dim = _unsqueeze_dims(tensor)

        assert not self.has_cls_embed

        if self.has_cls_embed:
            # TODO: enable has_cls_embed

            # cls_tok: Tensor = torch.tensor(0)  # For typing/torchscriptability
            # if self.has_cls_embed:
            #    cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]
            raise NotImplementedError("Unsupported the input tensor contains cls token")

        # input shape: B, num_heads, seqlen, head_dim
        B, N, L, C = get_shape(tensor)
        T, H, W = thw_shape
        tensor = ops.reshape()(tensor, [B * N, -1, H, W, C])

        if self.norm_before_pool:
            # If use BN, we apply norm before pooling instead of after pooling.
            tensor = self.norm(tensor)
            # We also empirically find that adding a GELU here is beneficial.
            tensor = ops.elementwise(FuncEnum.GELU)(tensor)

        tensor = self.pool(tensor)

        shape = get_shape(tensor)
        thw_shape = [shape[1], shape[2], shape[3]]
        L_pooled = shape[1] * shape[2] * shape[3]
        tensor = ops.reshape()(tensor, [B, N, L_pooled, C])

        if self.has_norm and not self.norm_before_pool:
            tensor = self.norm(tensor)

        tensor = _squeeze_dims(tensor, tensor_dim)

        return tensor, thw_shape


class MultiScaleAttention(Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.

    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    _version = 2

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        batch_size: int = 1,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer: Callable = LayerNorm,
        has_cls_embed: bool = True,
        pool_mode: str = "conv",
        pool_first: bool = False,
        residual_pool: bool = True,
        depthwise_conv: bool = True,
        bias_on: bool = True,
        separate_qkv: bool = False,
        max_seq_len: int = 6272,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """

        super().__init__()
        assert pool_mode in ["conv", "avg", "max"]

        self.pool_first = pool_first
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        self.max_seq_len = max_seq_len
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # Set placeholders for torchscriptability, may not be actually used
        self.q = self.k = self.v = self.qkv = Identity()
        if self.pool_first or self.separate_qkv:
            self.q = Linear(dim, dim, bias=qkv_bias)
            self.k = Linear(dim, dim, bias=qkv_bias)
            self.v = Linear(dim, dim, bias=qkv_bias)
        else:
            self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True if bias_on else False)

        assert dropout_rate == 0.0
        if dropout_rate > 0.0:
            self.proj_drop = Dropout(dropout_rate)
        else:
            self.proj_drop = Identity()

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            kernel_q is not None
            and self._prod(kernel_q) == 1
            and self._prod(stride_q) == 1
        ):
            kernel_q = None
        if (
            kernel_kv is not None
            and self._prod(kernel_kv) == 1
            and self._prod(stride_kv) == 1
        ):
            kernel_kv = None

        if pool_mode in ["max", "avg"]:
            raise NotImplementedError(f"Unsupported input dimension {pool_mode}")

        ## TODO: add pool mode support for {"max", "avg"}

        elif pool_mode == "conv":
            self.pool_q = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_q is not None
                else None
            )

            self.norm_q = norm_layer(head_dim) if kernel_q is not None else None
            self.pool_k = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_k = norm_layer(head_dim) if kernel_kv is not None else None
            self.pool_v = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )

            self.norm_v = norm_layer(head_dim) if kernel_kv is not None else None
        else:
            raise NotImplementedError(f"Unsupported model {pool_mode}")

        # Will not be used if `separate_qkv == True`
        self._attention_pool_q = _AttentionPool(
            self.pool_q,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        self._attention_pool_k = _AttentionPool(
            self.pool_k,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        self._attention_pool_v = _AttentionPool(
            self.pool_v,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

    def _qkv_proj(
        self,
        q: Tensor,
        q_size: int,
        k: Tensor,
        k_size: int,
        v: Tensor,
        v_size: int,
        batch_size: int,
        chan_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        q = ops.permute()(
            ops.reshape()(
                self.q(q)[
                    batch_size, q_size, self.num_heads, chan_size // self.num_heads
                ]
            ),
            [0, 2, 1, 3],
        )
        k = ops.permute()(
            ops.reshape()(
                self.k(k)[
                    batch_size, k_size, self.num_heads, chan_size // self.num_heads
                ]
            ),
            [0, 2, 1, 3],
        )
        v = ops.permute()(
            ops.reshape()(
                self.v(v)[
                    batch_size, v_size, self.num_heads, chan_size // self.num_heads
                ]
            ),
            [0, 2, 1, 3],
        )
        return q, k, v

    def _qkv_pool(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        thw_shape: List[int],
    ) -> Tuple[Tensor, List[int], Tensor, List[int], Tensor, List[int]]:
        q, q_shape = self._attention_pool_q(q, thw_shape)
        k, k_shape = self._attention_pool_k(k, thw_shape)
        v, v_shape = self._attention_pool_v(v, thw_shape)
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int, int, int]:
        q_N = self._prod(q_shape) + 1 if self.has_cls_embed else self._prod(q_shape)
        k_N = self._prod(k_shape) + 1 if self.has_cls_embed else self._prod(k_shape)
        v_N = self._prod(v_shape) + 1 if self.has_cls_embed else self._prod(v_shape)
        return q_N, k_N, v_N

    def _prod(self, shape: List[int]) -> int:
        """Torchscriptable version of `numpy.prod`. Note that `_prod([]) == 1`"""
        p: int = 1
        for dim in shape:
            p *= dim
        return p

    def _reshape_qkv_to_seq(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_N: int,
        v_N: int,
        k_N: int,
        B: int,
        C: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(self, x: Tensor, thw_shape: List[int]) -> Tuple[Tensor, List[int]]:
        """
        Args:
            x (Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        B, N, C = get_shape(x)
        if self.pool_first:
            x = ops.reshape()(x, [B, N, self.num_heads, C // self.num_heads])
            x = ops.permute()(x, [0, 2, 1, 3])
            q = k = v = x
            pass
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
            q_N, k_N, v_N = self._get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self._reshape_qkv_to_seq(q, k, v, q_N, v_N, k_N, B, C)
            q, k, v = self._qkv_proj(q, q_N, k, k_N, v, v_N, B, C)
        else:
            if self.separate_qkv:
                q = k = v = x
                pass
                # TODO: implement when separate_qkv
                # q, k, v = self._qkv_proj(q, N, k, N, v, N, B, C)
            else:
                # compute q, k, v and perform pooling
                qkv = ops.permute()(
                    ops.reshape()(self.qkv(x), [B, N, 3, self.num_heads, -1]),
                    [2, 0, 3, 1, 4],
                )
                # input shape: 3, B, num_heads, seqlen, head_dim
                shape = get_shape(qkv)
                # obtain q, k, v from qkv
                qkv = ops.reshape()(qkv, [3 * B, self.num_heads, N, shape[-1]])
                (q, k, v) = ops.split()(qkv, B, dim=0)
            q, q_thw_shape, k, k_thw_shape, v, v_thw_shape = self._qkv_pool(
                q, k, v, thw_shape
            )

        # attention
        q_shape = get_shape(q)
        B, num_heads, seqlen, head_dim = get_shape(q)
        score = ops.transpose()(ops.mem_eff_attention(causal=False)(q, k, v), 1, 2)

        if self.residual_pool:
            score = ops.elementwise(FuncEnum.ADD)(score, q)

        score = ops.reshape()(ops.transpose()(score, 1, 2), [B, -1, self.dim])

        score = self.proj(score)
        assert self.dropout_rate == 0.0
        if self.dropout_rate > 0.0:
            score = self.proj_drop(score)

        return score, q_thw_shape


class MultiScaleBlock(Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        droppath_rate: float = 0.0,
        act_layer: Module = GELU,
        norm_layer: Module = LayerNorm,
        attn_norm_layer: Module = LayerNorm,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        pool_mode: str = "conv",
        has_cls_embed: bool = True,
        pool_first: bool = False,
        residual_pool: bool = False,
        depthwise_conv: bool = True,
        bias_on: bool = True,
        separate_qkv: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (Module): Activation layer used in the Mlp layer.
            norm_layer (Module): Normalization layer.
            attn_norm_layer (Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.norm1_is_batchnorm_1d = isinstance(self.norm1, BatchNorm1d)
        self.norm1.permute_input_output = True if self.norm1_is_batchnorm_1d else False
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=attn_norm_layer,
            has_cls_embed=has_cls_embed,
            pool_mode=pool_mode,
            pool_first=pool_first,
            residual_pool=residual_pool,
            bias_on=bias_on,
            depthwise_conv=depthwise_conv,
            separate_qkv=separate_qkv,
            max_seq_len=seq_len,
        )
        self.drop_path = DropPath(droppath_rate) if droppath_rate > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_is_batchnorm_1d = isinstance(self.norm2, BatchNorm1d)
        self.norm2.permute_input_output = True if self.norm2_is_batchnorm_1d else False
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
            bias_on=bias_on,
        )
        if dim != dim_out:
            self.proj = Linear(dim, dim_out, bias=bias_on)
        else:
            self.proj = Identity()

        self.pool_skip = (
            MaxPool3d(tuple(kernel_skip), tuple(stride_skip), tuple(padding_skip))
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )
        self._attention_pool = _AttentionPool(
            self.pool_skip, has_cls_embed=self.has_cls_embed, norm=None
        )

    def forward(
        self, x: Tensor, t_shape: int, h_shape: int, w_shape: int
    ) -> Tuple[Tensor, List[int]]:
        """
        Args:
            x (Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """
        thw_shape = [t_shape, h_shape, w_shape]
        x_norm = (
            ait_ncl2nlc(self.norm1(ait_ncl2nlc(x)))
            if self.norm1_is_batchnorm_1d
            else self.norm1(x)
        )
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        x_res, _ = self._attention_pool(x, thw_shape)
        x = x_res + self.drop_path(x_block)
        x_norm = (
            ait_ncl2nlc(self.norm2(ait_ncl2nlc(x)))
            if self.norm2_is_batchnorm_1d
            else self.norm2(x)
        )
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        return x, thw_shape_new
