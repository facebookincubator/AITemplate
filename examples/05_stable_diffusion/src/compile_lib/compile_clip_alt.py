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
import sys

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
from .util import torch_dtype_from_str
import torch

USE_CUDA = detect_target().name() == "cuda"

def map_clip(pt_mod, device="cuda", dtype="float16"):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif USE_CUDA:
            if "q_proj" in name:
                ait_name = ait_name.replace("q_proj", "proj_q")
            elif "k_proj" in name:
                ait_name = ait_name.replace("k_proj", "proj_k")
            elif "v_proj" in name:
                ait_name = ait_name.replace("v_proj", "proj_v")
        else:
            if name.endswith("q_proj.weight"):
                ait_name = ait_name.replace("q_proj", "qkv")
                prefix = key[: -len("q_proj.weight")]
                q = pt_params[prefix + "q_proj.weight"]
                k = pt_params[prefix + "k_proj.weight"]
                v = pt_params[prefix + "v_proj.weight"]
                qkv_weight = torch.cat([q, k, v], dim=0).cuda()
                params_ait[ait_name] = qkv_weight
                continue
            elif name.endswith("q_proj.bias"):
                ait_name = ait_name.replace("q_proj", "qkv")
                prefix = key[: -len("q_proj.bias")]
                q = pt_params[prefix + "q_proj.bias"]
                k = pt_params[prefix + "k_proj.bias"]
                v = pt_params[prefix + "v_proj.bias"]
                qkv_bias = torch.cat([q, k, v], dim=0).cuda()
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
        # elif "q_proj" in name:
        #     ait_name = ait_name.replace("q_proj", "proj_q")
        # elif "k_proj" in name:
        #     ait_name = ait_name.replace("k_proj", "proj_k")
        # elif "v_proj" in name:
        #     ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr
    return params_ait


def compile_clip(
    pt_mod,
    batch_size=(1, 8),
    seqlen=64,
    dim=768,
    num_heads=12,
    depth=12,
    output_hidden_states=False,
    text_projection_dim=None,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    act_layer="gelu",
    constants=True,
    model_name="CLIPTextModel",
    work_dir="./tmp",
):
    mask_seq = 0
    causal = True

    pt_mod = pt_mod.eval()
    params_ait = map_clip(pt_mod)

    static_shape = batch_size[0] == batch_size[1]
    if static_shape:
        batch_size = batch_size[0]
    else:
        batch_size = IntVar(values=list(batch_size), name="batch_size")

    input_ids_ait = Tensor(
        [batch_size, seqlen], name="input_ids", dtype="int64", is_input=True
    )
    position_ids_ait = Tensor(
        [batch_size, seqlen], name="position_ids", dtype="int64", is_input=True
    )

    ait_mod = ait_CLIPTextTransformer(
        num_hidden_layers=depth,
        hidden_size=dim,
        num_attention_heads=num_heads,
        batch_size=batch_size,
        seq_len=seqlen,
        causal=causal,
        mask_seq=mask_seq,
        act_layer=act_layer,
        output_hidden_states=output_hidden_states,
        text_projection_dim=text_projection_dim,
    )
    ait_mod.name_parameter_tensor()
    
    Y = ait_mod(input_ids=input_ids_ait, position_ids=position_ids_ait)
    for out in Y:
        shape = [d._attrs["values"] for d in out._attrs["shape"]]
        print(f'AIT {out._attrs["name"]} shape: {shape}')

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    dll_name = model_name + ".dll" if sys.platform == "win32" else model_name + ".so"
    compile_model(
        Y,
        target,
        work_dir,
        model_name,
        constants=params_ait if constants else None,
        dll_name=dll_name,
    )
