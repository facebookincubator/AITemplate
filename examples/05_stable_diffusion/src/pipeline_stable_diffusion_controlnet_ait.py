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
import inspect
import os
from typing import List, Optional, Union

import torch
from aitemplate.compiler import Model
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.pil_utils import numpy_to_pil
from tqdm import tqdm
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from .compile_lib.compile_vae_alt import map_vae


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, additional_replacements=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(
        paths, list
    ), "Paths should be a list of dicts containing 'old' and 'new' keys."

    for path in paths:
        new_path = path["new"]

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


# ================#
# VAE Conversion #
# ================#


def convert_ldm_vae_checkpoint(vae_state_dict):
    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"
    ]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"
    ]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"
    ]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"
    ]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"
    ]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"
    ]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path]
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


# =================#
# UNet Conversion #
# =================#
def convert_ldm_unet_checkpoint(unet_state_dict, layers_per_block=2):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict[
        "time_embed.0.weight"
    ]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict[
        "time_embed.0.bias"
    ]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict[
        "time_embed.2.weight"
    ]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict[
        "time_embed.2.bias"
    ]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "input_blocks" in layer
        }
    )
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "middle_block" in layer
        }
    )
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "output_blocks" in layer
        }
    )
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (layers_per_block + 1)
        layer_in_block_id = (i - 1) % (layers_per_block + 1)

        resnets = [
            key
            for key in input_blocks[i]
            if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.weight"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.weight")
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.bias"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path]
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {
                "old": f"input_blocks.{i}.1",
                "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths,
        new_checkpoint,
        unet_state_dict,
        additional_replacements=[meta_path],
    )

    for i in range(num_output_blocks):
        block_id = i // (layers_per_block + 1)
        layer_in_block_id = i % (layers_per_block + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [
                key for key in output_blocks[i] if f"output_blocks.{i}.1" in key
            ]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(
                    ["conv.bias", "conv.weight"]
                )
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.weight"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.weight"]
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.bias"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.bias"]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths,
                    new_checkpoint,
                    unet_state_dict,
                    additional_replacements=[meta_path],
                )
        else:
            resnet_0_paths = renew_resnet_paths(
                output_block_layers, n_shave_prefix_segments=1
            )
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(
                    [
                        "up_blocks",
                        str(block_id),
                        "resnets",
                        str(layer_in_block_id),
                        path["new"],
                    ]
                )

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


# =========================#
#    AITemplate mapping   #
# =========================#
def map_unet_state_dict(state_dict, dim=320):
    params_ait = {}
    for key, arr in state_dict.items():
        arr = arr.to("cuda", dtype=torch.float16)
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            # print("ff.net.0.proj.weight")
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            # print("ff.net.0.proj.bias")
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def map_clip_state_dict(state_dict):
    params_ait = {}
    for key, arr in state_dict.items():
        arr = arr.to("cuda", dtype=torch.float16)
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr

    return params_ait


def map_controlnet_params(pt_mod):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr
    params_ait["controlnet_cond_embedding_conv_in_weight"] = torch.nn.functional.pad(
        params_ait["controlnet_cond_embedding_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
    )
    params_ait["arange"] = (
        torch.arange(start=0, end=320 // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


class StableDiffusionAITPipeline:
    def __init__(self, hf_hub_or_path, ckpt):
        self.device = torch.device("cuda")
        workdir = "tmp/"
        state_dict = None
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location="cpu")
            while "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            clip_state_dict = {}
            unet_state_dict = {}
            vae_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("cond_stage_model.transformer."):
                    new_key = key.replace("cond_stage_model.transformer.", "")
                    clip_state_dict[new_key] = state_dict[key]
                elif key.startswith("cond_stage_model.model."):
                    new_key = key.replace("cond_stage_model.model.", "")
                    clip_state_dict[new_key] = state_dict[key]
                elif key.startswith("first_stage_model."):
                    new_key = key.replace("first_stage_model.", "")
                    vae_state_dict[new_key] = state_dict[key]
                elif key.startswith("model.diffusion_model."):
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_state_dict[new_key] = state_dict[key]
            # TODO: SD2.x clip support, get from diffusers convert_from_ckpt.py
            # clip_state_dict = convert_text_enc_state_dict(clip_state_dict)
            unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict)
            vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict)
            state_dict = None

        self.controlnet_ait_exe = self.init_ait_module("ControlNetModel", "./tmp")
        print("Loading PyTorch ControlNet")
        controlnet_pt = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        ).to("cuda")
        controlnet_pt.eval()
        ait_params = map_controlnet_params(controlnet_pt)
        self.controlnet_ait_exe.set_many_constants_with_tensors(ait_params)
        self.controlnet_ait_exe.fold_constants()
        self.clip_ait_exe = self.init_ait_module(
            model_name="CLIPTextModel", workdir=workdir
        )
        print("Loading PyTorch CLIP")
        if ckpt is None:
            self.clip_pt = CLIPTextModel.from_pretrained(
                hf_hub_or_path,
                subfolder="text_encoder",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
        else:
            config = CLIPTextConfig.from_pretrained(
                hf_hub_or_path, subfolder="text_encoder"
            )
            self.clip_pt = CLIPTextModel(config)
            self.clip_pt.load_state_dict(clip_state_dict)
        clip_params_ait = map_clip_state_dict(dict(self.clip_pt.named_parameters()))
        print("Setting constants")
        self.clip_ait_exe.set_many_constants_with_tensors(clip_params_ait)
        print("Folding constants")
        self.clip_ait_exe.fold_constants()
        # cleanup
        self.clip_pt = None
        clip_params_ait = None

        self.unet_ait_exe = self.init_ait_module(
            model_name="ControlNetUNet2DConditionModel", workdir=workdir
        )

        print("Loading PyTorch UNet")
        if ckpt is None:
            self.unet_pt = UNet2DConditionModel.from_pretrained(
                hf_hub_or_path,
                subfolder="unet",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
            self.unet_pt = self.unet_pt.state_dict()
        else:
            self.unet_pt = unet_state_dict
        unet_params_ait = map_unet_state_dict(self.unet_pt)
        print("Setting constants")
        self.unet_ait_exe.set_many_constants_with_tensors(unet_params_ait)
        print("Folding constants")
        self.unet_ait_exe.fold_constants()
        # cleanup
        self.unet_pt = None
        unet_params_ait = None

        self.vae_ait_exe = self.init_ait_module(
            model_name="AutoencoderKL", workdir=workdir
        )
        print("Loading PyTorch VAE")
        if ckpt is None:
            self.vae_pt = AutoencoderKL.from_pretrained(
                hf_hub_or_path,
                subfolder="vae",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
        else:
            self.vae_pt = dict(vae_state_dict)

        print("Mapping parameters...")
        vae_params_ait = map_vae(self.vae_pt)
        print("Setting constants")
        self.vae_ait_exe.set_many_constants_with_tensors(vae_params_ait)
        print("Folding constants")
        self.vae_ait_exe.fold_constants()
        # cleanup
        self.vae_pt = None
        vae_params_ait = None

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.batch = 1

    def init_ait_module(
        self,
        model_name,
        workdir,
    ):
        mod = Model(os.path.join(workdir, model_name, "test.so"))
        return mod

    def controlnet_inference(
        self, latent_model_input, timesteps, encoder_hidden_states, controlnet_cond
    ):
        exe_module = self.controlnet_ait_exe
        timesteps_pt = timesteps.expand(latent_model_input.shape[0])
        inputs = {
            "input0": latent_model_input.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
            "input3": controlnet_cond.permute((0, 2, 3, 1)).contiguous().cuda().half(),
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        down_block_residuals = (y for y in ys[:-1])
        mid_block_residuals = ys[-1]
        return down_block_residuals, mid_block_residuals

    def unet_inference(
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        height,
        width,
        down_block_residuals,
        mid_block_residual,
    ):
        exe_module = self.unet_ait_exe
        timesteps_pt = timesteps.expand(self.batch * 2)
        inputs = {
            "input0": latent_model_input.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        for i, y in enumerate(down_block_residuals):
            inputs[f"down_block_residual_{i}"] = y
        inputs["mid_block_residual"] = mid_block_residual
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch * 2
            shape[1] = height // 8
            shape[2] = width // 8
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=77):
        exe_module = self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input, height, width):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch * 2
            shape[1] = height
            shape[2] = width
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        control_cond: torch.FloatTensor,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined  as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        self.batch = batch_size

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))
        # pytorch equivalent
        # text_embeddings = self.clip_pt(text_input.input_ids.to(self.device)).last_hidden_state

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = self.device
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            down_block_residuals, mid_block_residual = self.controlnet_inference(
                latent_model_input, t, text_embeddings, control_cond
            )
            # predict the noise residual
            noise_pred = self.unet_inference(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                height=height,
                width=width,
                down_block_residuals=down_block_residuals,
                mid_block_residual=mid_block_residual,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_inference(latents, height, width)
        # pytorch equivalent
        # image = self.vae_pt.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        has_nsfw_concept = None

        if output_type == "pil":
            image = numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
