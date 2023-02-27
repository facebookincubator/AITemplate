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
from typing import Any, Dict, OrderedDict, Tuple

import numpy as np

import torch
from aitemplate.backend.target import Target
from aitemplate.compiler.base import _TorchConstantTensorData
from aitemplate.testing import detect_target
from aitemplate.frontend import nn
from torch.fx.node import Argument

from .ait_converters import ConverterOutput
from .converter_registry import ait_converter

USE_CUDA = detect_target().name() == "cuda"


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
    if USE_CUDA:
        attn = nn.CrossAttention(
            dim=submod.embed_dim,
            seq_len=seq_len_q.value(),
            seq_len_kv=seq_len.value(),
            num_heads=submod.num_heads,
            qkv_bias=True,
            has_residual=False,
        )
    else:
        attn = nn.MultiheadAttention(
            dim=submod.embed_dim,
            batch_size=bsz.value(),
            seq_len=seq_len_q.value(),
            num_heads=submod.num_heads,
            qkv_bias=True,
            has_residual=False,
            use_mem_eff=True
        )

    # Bind constant tensor for MHA module
    mapped_params = _map_ait_pt_params(attn, submod)
    ait_params = dict(attn.named_parameters())
    for name, data in mapped_params.items():
        ait_tensor = ait_params[name].tensor()
        ait_data = _TorchConstantTensorData(data.contiguous().cuda().half())
        ait_tensor._bind_data(ait_data)

    if "cu_length" in ait_params:
        ait_tensor = ait_params["cu_length"].tensor()
        cu_len = np.cumsum([0] + [seq_len.value()] * bsz.value()).astype("int32")
        cu_len = torch.from_numpy(cu_len)
        ait_data = _TorchConstantTensorData(cu_len.contiguous().cuda())
        ait_tensor._bind_data(ait_data)

    res = attn(query, key, value)
    # make output of MHA a list to match the output type of pytorch MHA
    return [res]


def _map_ait_pt_params(ait_module, pt_module):
    ait_params = dict(ait_module.named_parameters())
    mapped_pt_params = OrderedDict()
    for pt_name, pt_param in pt_module.named_parameters():
        ait_friendly_name = (
            pt_name.replace("in_proj", "qkv")
            .replace("out_proj", "proj")
            .replace("_", ".")
        )
        if ait_friendly_name in ait_params:
            mapped_pt_params[ait_friendly_name] = pt_param.data
        elif "in_proj" in pt_name:
            # set constant for cross attention
            if len(pt_param.shape) == 2:
                w_q, w_k, w_v = pt_param.chunk(3)
                mapped_pt_params["proj_q.weight"] = w_q
                mapped_pt_params["proj_k.weight"] = w_k
                mapped_pt_params["proj_v.weight"] = w_v
            else:
                b_q, b_k, b_v = pt_param.chunk(3)
                mapped_pt_params["proj_q.bias"] = b_q
                mapped_pt_params["proj_k.bias"] = b_k
                mapped_pt_params["proj_v.bias"] = b_v
    return mapped_pt_params
