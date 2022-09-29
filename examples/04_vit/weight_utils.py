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
"""script for converting vit model from timm to ait
"""
import pickle

import click
import torch
import torch.nn as nn
from aitemplate.testing.detect_target import detect_target
from timm.models.vision_transformer import (
    VisionTransformer,
    vit_base_patch16_224,
    vit_large_patch16_384,
)


def convert_vit(model_name, pretrained=False):
    img_size = 224
    embed_dim = 768
    patch_size = 16
    depth = 12
    mod = None
    if model_name == "vit_base_patch16_224":
        if pretrained:
            mod = vit_base_patch16_224(pretrained=pretrained).cuda().half()
        else:
            mod = (
                VisionTransformer(
                    img_size=img_size,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    class_token=False,
                    global_pool="avg",
                    depth=depth,
                    patch_size=patch_size,
                    num_heads=12,
                    embed_dim=embed_dim,
                )
                .cuda()
                .half()
            )
    elif model_name == "vit_large_patch16_384":
        img_size = 384
        embed_dim = 1024
        depth = 24
        if pretrained:
            mod = vit_large_patch16_384(pretrained=pretrained).cuda().half()
        else:
            mod = (
                VisionTransformer(
                    img_size=img_size,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    class_token=False,
                    global_pool="avg",
                    depth=24,
                    patch_size=patch_size,
                    num_heads=16,
                    embed_dim=embed_dim,
                )
                .cuda()
                .half()
            )
    else:
        print(model_name)
        raise NotImplementedError
    params_pt = mod.named_parameters()
    params_ait = {}
    params_ait = {}
    for key, arr in params_pt:
        ait_key = key.replace(".", "_")
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
            if detect_target().name() == "cuda":
                conv0_w_pad = (
                    torch.zeros((embed_dim, patch_size, patch_size, 4)).cuda().half()
                )
                conv0_w_pad[:, :, :, :3] = arr
                arr = conv0_w_pad
        params_ait[f"{ait_key}"] = arr
    return params_ait


def export_to_torch_tensor(model_name, pretrained=False):
    params_ait = convert_vit(model_name, pretrained)
    return params_ait


@click.command()
@click.option("--model_name", default="vit_base_patch16_224", help="model name")
@click.option("--param-path", default="vit.pkl", help="saved numpy weights path")
@click.option("--pretrained", default=False, help="use pretrained weights")
def export_to_numpy(model_name, param_path, pretrained=False):
    params_ait = convert_vit(model_name, pretrained)
    params_np = {k: v.detach().cpu().numpy() for k, v in params_ait.items()}

    with open(param_path, "wb") as f:
        pickle.dump(params_np, f)


if __name__ == "__main__":
    export_to_numpy()
