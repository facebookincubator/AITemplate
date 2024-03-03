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
from functools import partial

from aitemplate.compiler import ops
from aitemplate.frontend import nn
from aitemplate.testing import detect_target

# pylint: disable=W0102

USE_CUDA = detect_target().name() == "cuda"


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            specialization="fast_gelu" if act_layer == "GELU" else "relu",
        )
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        shape = get_shape(x)
        x = self.fc1(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, shape)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        batch_size,
        seq_len,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            batch_size,
            seq_len,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.attn(self.norm1(x), x)
        x = self.mlp(self.norm2(x), x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        conv_op = (
            nn.Conv2dBiasFewChannels
            if detect_target().name() == "cuda"
            else nn.Conv2dBias
        )
        self.proj = conv_op(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj_norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, H, W, C = get_shape(x)
        x = self.proj(x)
        if self.flatten:
            x = ops.reshape()(x, [B, -1, self.embed_dim])
        x = self.proj_norm(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        batch_size=1,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=nn.LayerNorm,
        act_layer=None,
        block_fn=Block,
        dtype="float16",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(shape=[1, 1, embed_dim], dtype=dtype) if class_token else None
        )
        self.cls_token_mask = (
            nn.Parameter(shape=[batch_size, 1, embed_dim], dtype=dtype)
            if class_token
            else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(shape=[1, embed_len, embed_dim], dtype=dtype)
        self.pos_drop = nn.Dropout(p=drop_rate)
        seq_len = (img_size // patch_size) ** 2 + (1 if class_token else 0)
        self.pool_size = img_size // patch_size

        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        if global_pool == "avg":
            self.pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=1, padding=0)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed.tensor()
            if self.cls_token is not None:
                cls_token_expand = ops.expand()(
                    self.cls_token.tensor(), [get_shape(x)[0], -1, -1]
                )
                cls_token_expand = cls_token_expand + self.cls_token_mask.tensor()
                x = ops.concatenate()([cls_token_expand, x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                cls_token_expand = ops.expand()(
                    self.cls_token.tensor(), [get_shape(x)[0], -1, -1]
                )
                cls_token_expand = cls_token_expand + self.cls_token_mask.tensor()
                x = ops.concatenate()([cls_token_expand, x], dim=1)
            x = x + self.pos_embed.tensor()
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def _global_pool(self, x):
        batch, seq, d = get_shape(x)
        x = ops.reshape()(x, [batch, self.pool_size, self.pool_size, d])
        y = self.pool(x)
        return ops.reshape()(y, [batch, d])

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            if self.global_pool == "avg":
                x = self._global_pool(x)
            else:
                batch, seq, d = get_shape(x)
                x = ops.dynamic_slice()(
                    x, start_indices=[0, 0, 0], end_indices=[batch, 1, d]
                )
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
