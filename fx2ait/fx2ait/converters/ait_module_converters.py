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
from typing import Any, Dict, Tuple

import numpy as np

import torch
from aitemplate.backend.target import Target
from aitemplate.compiler.base import _TorchConstantTensorData, Tensor
from aitemplate.compiler.ops import chunk
from aitemplate.frontend import nn
from fx2ait.utils import make_str_ait_friendly
from torch.fx.node import Argument

from .ait_converters import ConverterOutput
from .converter_registry import ait_converter


@ait_converter(torch.nn.modules.activation.MultiheadAttention)
def multi_head_attention_module(
    target: Target,
    submod: Any,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO fix arg/kwargs matching
    query = kwargs["query"] if "query" in kwargs else args[0]
    key = kwargs["key"] if "key" in kwargs else args[1]
    value = kwargs["value"] if "value" in kwargs else args[2]
    bsz, seq_len_q, dim = query.shape()
    _, seq_len, _ = key.shape()

    assert (
        submod.embed_dim % submod.num_heads == 0
    ), f"embed_dim {submod.embed_dim} must be divisible by num_heads {submod.num_heads}"
    head_size = submod.embed_dim // submod.num_heads
    if head_size % 4 != 0:
        raise ValueError(
            f"The head size {head_size} (ie. embed_dim ({submod.embed_dim}) / num_heads ({submod.num_heads}) "
            " must be divisible by 4. Please fix the model or consider using the complete_video_view_all_page_types preset",
        )

    attn = nn.CrossAttention(
        dim=submod.embed_dim,
        seq_len=seq_len_q.value(),
        seq_len_kv=seq_len.value(),
        num_heads=submod.num_heads,
        qkv_bias=True,
        has_residual=False,
    )

    # Bind constant tensor for MHA module
    qkv_weight, qkv_bias = None, None
    for k, v in submod.named_parameters():
        ait_data = _TorchConstantTensorData(v.data.contiguous().cuda().half())
        if "in_proj" in k:
            if "weight" in k:
                qkv_weight = Tensor(
                    shape=v.shape,
                    dtype="float16",
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                qkv_weight._bind_data(ait_data)
            elif "bias" in k:
                qkv_bias = Tensor(
                    shape=v.shape,
                    dtype="float16",
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                qkv_bias._bind_data(ait_data)
        elif "out_proj" in k:
            if "weight" in k:
                tensor = attn.proj.weight.tensor()
            elif "bias" in k:
                tensor = attn.proj.bias.tensor()
            tensor._attrs["name"] = make_str_ait_friendly(f"{target}.{k}")
            tensor._bind_data(ait_data)

    # Swap out qkv tensor used by nn.CrossAttention.
    q_w, k_w, v_w = chunk()(qkv_weight, 3)
    q_b, k_b, v_b = chunk()(qkv_bias, 3)

    attn.proj_q.weight._tensor = q_w
    attn.proj_k.weight._tensor = k_w
    attn.proj_v.weight._tensor = v_w
    attn.proj_q.bias._tensor = q_b
    attn.proj_k.bias._tensor = k_b
    attn.proj_v.bias._tensor = v_b

    ait_params = dict(attn.named_parameters())
    if "cu_length" in ait_params:
        ait_tensor = ait_params["cu_length"].tensor()
        cu_len = np.cumsum([0] + [seq_len.value()] * bsz.value()).astype("int32")
        cu_len = torch.from_numpy(cu_len)
        ait_data = _TorchConstantTensorData(cu_len.contiguous().cuda())
        ait_tensor._bind_data(ait_data)

    res = attn(query, key, value)
    # make output of MHA a list to match the output type of pytorch MHA
    return [res]
