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
import numpy as np
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from ..modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
from .util import mark_output


def map_clip_params(pt_mod, batch_size, seqlen, depth):

    params_pt = list(pt_mod.named_parameters())

    params_ait = {}
    pt_params = {}
    for key, arr in params_pt:
        pt_params[key.replace("text_model.", "")] = arr

    pt_params = dict(pt_mod.named_parameters())
    for key, arr in pt_params.items():
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("q_proj.weight"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.weight")]
            q = pt_params[prefix + "q_proj.weight"]
            k = pt_params[prefix + "k_proj.weight"]
            v = pt_params[prefix + "v_proj.weight"]
            qkv_weight = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_weight
            continue
        elif name.endswith("q_proj.bias"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.bias")]
            q = pt_params[prefix + "q_proj.bias"]
            k = pt_params[prefix + "k_proj.bias"]
            v = pt_params[prefix + "v_proj.bias"]
            qkv_bias = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_bias
            continue
        elif name.endswith("k_proj.weight"):
            continue
        elif name.endswith("k_proj.bias"):
            continue
        elif name.endswith("v_proj.weight"):
            continue
        elif name.endswith("v_proj.bias"):
            continue
        params_ait[ait_name] = arr

        if detect_target().name() == "cuda":
            for i in range(depth):
                prefix = "encoder_layers_%d_self_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait


def compile_clip(
    pt_mod,
    batch_size=1,
    seqlen=64,
    dim=768,
    num_heads=12,
    depth=12,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    act_layer="gelu",
):
    mask_seq = 0
    causal = True

    ait_mod = ait_CLIPTextTransformer(
        num_hidden_layers=depth,
        hidden_size=dim,
        num_attention_heads=num_heads,
        batch_size=batch_size,
        seq_len=seqlen,
        causal=causal,
        mask_seq=mask_seq,
        act_layer=act_layer,
    )
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_clip_params(pt_mod, batch_size, seqlen, depth)

    input_ids_ait = Tensor(
        [batch_size, seqlen], name="input0", dtype="int64", is_input=True
    )
    position_ids_ait = Tensor(
        [batch_size, seqlen], name="input1", dtype="int64", is_input=True
    )
    Y = ait_mod(input_ids=input_ids_ait, position_ids=position_ids_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, "./tmp", "CLIPTextModel", constants=params_ait)
