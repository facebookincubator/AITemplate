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

from ..modeling.vae import AutoencoderKL as ait_AutoencoderKL
from .util import mark_output


def map_vae_params(ait_module, pt_module, batch_size, seq_len):
    pt_params = dict(pt_module.named_parameters())
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
        elif name.endswith("attention.qkv.weight"):
            prefix = name[: -len("attention.qkv.weight")]
            q_weight = pt_params[prefix + "query.weight"]
            k_weight = pt_params[prefix + "key.weight"]
            v_weight = pt_params[prefix + "value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_pt_params[ait_name] = qkv_weight
        elif name.endswith("attention.qkv.bias"):
            prefix = name[: -len("attention.qkv.bias")]
            q_bias = pt_params[prefix + "query.bias"]
            k_bias = pt_params[prefix + "key.bias"]
            v_bias = pt_params[prefix + "value.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            mapped_pt_params[ait_name] = qkv_bias
        elif name.endswith("attention.proj.weight"):
            prefix = name[: -len("attention.proj.weight")]
            pt_name = prefix + "proj_attn.weight"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj.bias"):
            prefix = name[: -len("attention.proj.bias")]
            pt_name = prefix + "proj_attn.bias"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.cu_length"):
            cu_len = np.cumsum([0] + [seq_len] * batch_size).astype("int32")
            mapped_pt_params[ait_name] = torch.from_numpy(cu_len).cuda()
        else:
            pt_param = pt_module.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param

    return mapped_pt_params


def compile_vae(
    pt_mod,
    batch_size=1,
    height=64,
    width=64,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
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

    ait_vae = ait_AutoencoderKL(
        batch_size,
        height,
        width,
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
    ait_input = Tensor(
        shape=[batch_size, height, width, latent_channels],
        name="vae_input",
        is_input=True,
    )
    ait_vae.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_vae_params(ait_vae, pt_mod, batch_size, height * width)

    Y = ait_vae.decode(ait_input)
    mark_output(Y)
    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        "./tmp",
        "AutoencoderKL",
        constants=params_ait,
    )
