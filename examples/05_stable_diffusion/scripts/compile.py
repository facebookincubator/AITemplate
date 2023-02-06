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
import logging

import click

import torch

from aitemplate.testing import detect_target
from aitemplate.utils.import_path import import_parent

from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.compile_lib.compile_clip import compile_clip
from src.compile_lib.compile_unet import compile_unet
from src.compile_lib.compile_vae import compile_vae


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="the local diffusers pipeline directory",
)
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option("--batch-size", default=1, help="batch size")
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def compile_diffusers(
    local_dir, width, height, batch_size, use_fp16_acc=True, convert_conv_to_gemm=True
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = StableDiffusionPipeline.from_pretrained(
        local_dir,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    ww = width // 8
    hh = height // 8

    # CLIP
    compile_clip(
        pipe.text_encoder,
        batch_size=batch_size,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        depth=pipe.text_encoder.config.num_hidden_layers,
        num_heads=pipe.text_encoder.config.num_attention_heads,
        dim=pipe.text_encoder.config.hidden_size,
        act_layer=pipe.text_encoder.config.hidden_act,
    )
    # UNet
    compile_unet(
        pipe.unet,
        batch_size=batch_size * 2,
        width=ww,
        height=hh,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        hidden_dim=pipe.unet.config.cross_attention_dim,
        attention_head_dim=pipe.unet.config.attention_head_dim,
    )
    # VAE
    compile_vae(
        pipe.vae,
        batch_size=batch_size,
        width=ww,
        height=hh,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
    )


if __name__ == "__main__":
    compile_diffusers()
