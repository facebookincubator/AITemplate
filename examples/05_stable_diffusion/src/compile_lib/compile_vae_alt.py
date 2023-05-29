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

import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.vae import AutoencoderKL as ait_AutoencoderKL
from .util import mark_output


def map_vae_params(ait_module, pt_module, batch_size=1, seq_len=4096):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    mapped_pt_params = {}
    for name, _ in ait_module.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            if (
                "conv" in name
                and "norm" not in name
                and name.endswith(".weight")
                and len(pt_params[name].shape) == 4
            ):
                mapped_pt_params[ait_name] = torch.permute(
                    pt_params[name], [0, 2, 3, 1]
                ).contiguous()
            else:
                mapped_pt_params[ait_name] = pt_params[name]
        elif name.endswith("attention.proj.weight"):
            prefix = name[: -len("attention.proj.weight")]
            pt_name = prefix + "proj_attn.weight"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_out.0.weight"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj.bias"):
            prefix = name[: -len("attention.proj.bias")]
            pt_name = prefix + "proj_attn.bias"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_out.0.bias"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.cu_length"):
            ...
        elif name.endswith("attention.proj_q.weight"):
            prefix = name[: -len("attention.proj_q.weight")]
            pt_name = prefix + "query.weight"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_q.weight"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_q.bias"):
            prefix = name[: -len("attention.proj_q.bias")]
            pt_name = prefix + "query.bias"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_q.bias"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_k.weight"):
            prefix = name[: -len("attention.proj_k.weight")]
            pt_name = prefix + "key.weight"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_k.weight"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_k.bias"):
            prefix = name[: -len("attention.proj_k.bias")]
            pt_name = prefix + "key.bias"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_k.bias"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_v.weight"):
            prefix = name[: -len("attention.proj_v.weight")]
            pt_name = prefix + "value.weight"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_v.weight"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_v.bias"):
            prefix = name[: -len("attention.proj_v.bias")]
            pt_name = prefix + "value.bias"
            if pt_name in pt_params:
                mapped_pt_params[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_v.bias"
                mapped_pt_params[ait_name] = pt_params[pt_name]
        else:
            pt_param = pt_module.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param
    for key, arr in mapped_pt_params.items():
        mapped_pt_params[key] = arr.to("cuda", dtype=torch.float16)
    return mapped_pt_params


def compile_vae(
    pt_mod,
    batch_size=(1, 8),
    height=(64, 2048),
    width=(64, 2048),
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    name="AutoencoderKL",
    constants=True,
):
    in_channels = 3
    out_channels = 3
    down_block_types = [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ]
    up_block_types = [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ]
    block_out_channels = [128, 256, 512, 512]
    layers_per_block = 2
    act_fn = "silu"
    latent_channels = 4
    sample_size = 512

    # values not important, we only need this for mapping keys
    ait_vae = ait_AutoencoderKL(
        1,
        64,
        64,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        act_fn=act_fn,
        latent_channels=latent_channels,
        sample_size=sample_size,
    )
    batch_size = IntVar(values=list(batch_size), name="batch_size")
    height = height[0] // 8, height[1] // 8
    width = width[0] // 8, width[1] // 8
    height_d = IntVar(values=list(height), name="height")
    width_d = IntVar(values=list(width), name="width")

    ait_input = Tensor(
        shape=[batch_size, height_d, width_d, latent_channels],
        name="vae_input",
        is_input=True,
    )
    ait_vae.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_vae_params(ait_vae, pt_mod)

    Y = ait_vae.decode(ait_input)
    mark_output(Y)
    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        "./tmp",
        name,
        constants=params_ait if constants else None,
    )
