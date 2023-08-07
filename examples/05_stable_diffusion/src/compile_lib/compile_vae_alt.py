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

import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.vae import AutoencoderKL as ait_AutoencoderKL


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def map_vae(pt_module, device="cuda", dtype="float16", encoder=False):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_ait = {}
    quant_key = "post_quant" if encoder else "quant"
    vae_key = "decoder" if encoder else "encoder"
    for key, arr in pt_params.items():
        if key.startswith(vae_key):
            continue
        if key.startswith(quant_key):
            continue
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            params_ait[key] = torch.permute(arr, [0, 2, 3, 1]).contiguous()
        elif key.endswith("proj_attn_weight"):
            prefix = key[: -len("proj_attn_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("to_out_0_weight"):
            prefix = key[: -len("to_out_0_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("proj_attn_bias"):
            prefix = key[: -len("proj_attn_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("to_out_0_bias"):
            prefix = key[: -len("to_out_0_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("query_weight"):
            prefix = key[: -len("query_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("to_q_weight"):
            prefix = key[: -len("to_q_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("query_bias"):
            prefix = key[: -len("query_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("to_q_bias"):
            prefix = key[: -len("to_q_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("key_weight"):
            prefix = key[: -len("key_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("key_bias"):
            prefix = key[: -len("key_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("value_weight"):
            prefix = key[: -len("value_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("value_bias"):
            prefix = key[: -len("value_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        elif key.endswith("to_k_weight"):
            prefix = key[: -len("to_k_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("to_v_weight"):
            prefix = key[: -len("to_v_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("to_k_bias"):
            prefix = key[: -len("to_k_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("to_v_bias"):
            prefix = key[: -len("to_v_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        else:
            params_ait[key] = arr
    if encoder:
        params_ait["encoder_conv_in_weight"] = torch.functional.F.pad(
            params_ait["encoder_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
        )

    return params_ait


def compile_vae(
    pt_mod,
    batch_size=(1, 8),
    height=(64, 2048),
    width=(64, 2048),
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    model_name="AutoencoderKL",
    constants=True,
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    act_fn="silu",
    latent_channels=4,
    sample_size=512,
    in_channels=3,
    out_channels=3,
    down_block_types=[
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    up_block_types=[
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    input_size=(64, 64),
    down_factor=8,
    dtype="float16",
    work_dir="./tmp",
    vae_encode=False,
):
    ait_vae = ait_AutoencoderKL(
        batch_size[0],
        input_size[0],
        input_size[1],
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        act_fn=act_fn,
        latent_channels=latent_channels,
        sample_size=sample_size,
        dtype=dtype,
    )

    static_batch = batch_size[0] == batch_size[1]
    static_shape = height[0] == height[1] and width[0] == width[1]
    if not vae_encode:
        height = height[0] // down_factor, height[1] // down_factor
        width = width[0] // down_factor, width[1] // down_factor

    if static_batch:
        batch_size = batch_size[0]
    else:
        batch_size = IntVar(values=list(batch_size), name="batch_size")
    if static_shape:
        height_d = height[0]
        width_d = width[0]
    else:
        height_d = IntVar(values=list(height), name="height")
        width_d = IntVar(values=list(width), name="width")

    ait_input = Tensor(
        shape=[batch_size, height_d, width_d, 3 if vae_encode else latent_channels],
        name="pixels" if vae_encode else "latent",
        is_input=True,
        dtype=dtype,
    )
    sample = None
    if vae_encode:
        sample = Tensor(
            shape=[batch_size, height_d, width_d, latent_channels],
            name="random_sample",
            is_input=True,
            dtype=dtype,
        )
    ait_vae.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_vae(pt_mod, dtype=dtype, encoder=vae_encode)
    if vae_encode:
        Y = ait_vae.encode(ait_input, sample)
    else:
        Y = ait_vae.decode(ait_input)
    shape = [d._attrs["values"] for d in Y._attrs["shape"]]
    print(f'AIT {Y._attrs["name"]} shape: {shape}')
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
