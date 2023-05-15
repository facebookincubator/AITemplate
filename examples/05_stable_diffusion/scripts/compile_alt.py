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

from src.compile_lib.compile_clip_alt import compile_clip
from src.compile_lib.compile_unet_alt import compile_unet
from src.compile_lib.compile_vae_alt import compile_vae


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/runwayml/stable-diffusion-v1-5",
    help="the local diffusers pipeline directory",
)
@click.option("--min-width", default=512, help="Minimum width of generated image")
@click.option("--max-width", default=512, help="Maximum width of generated image")
@click.option("--min-height", default=512, help="Minimum height of generated image")
@click.option("--max-height", default=512, help="Maximum height of generated image")
@click.option("--batch-size", default=1, help="batch size")
@click.option("--clip-chunks", default=1, help="batch size")
@click.option("--include-constants", default=None, help="include constants (model weights) with compiled model")
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def compile_diffusers(
    local_dir, min_width, max_width, min_height, max_height, batch_size, clip_chunks, include_constants, use_fp16_acc=True, convert_conv_to_gemm=True, 
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

    assert (
        min_width % 64 == 0 and max_width % 64 == 0
    ), "Minimum Width and Maximum Width must be multiples of 64, otherwise, the compilation process will fail."
    assert (
        min_height % 64 == 0 and max_height % 64 == 0
    ), "Minimum Height and Maximum Height must be multiples of 64, otherwise, the compilation process will fail."

    min_height=min_height // 8
    max_height=max_height // 8
    min_width=min_width // 8
    max_width=max_width // 8
    # CLIP
    compile_clip(
        pipe.text_encoder,
        batch_size=batch_size,
        seqlen=77,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        depth=pipe.text_encoder.config.num_hidden_layers,
        num_heads=pipe.text_encoder.config.num_attention_heads,
        dim=pipe.text_encoder.config.hidden_size,
        act_layer=pipe.text_encoder.config.hidden_act,
        constants=True if include_constants else False,
    )
    # UNet
    compile_unet(
        pipe.unet,
        batch_size=batch_size * 2,
        min_height=min_height,
        max_height=max_height,
        min_width=min_width,
        max_width=max_width,
        clip_chunks=clip_chunks,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        hidden_dim=pipe.unet.config.cross_attention_dim,
        attention_head_dim=pipe.unet.config.attention_head_dim,
        use_linear_projection=pipe.unet.config.get("use_linear_projection", False),
        constants=True if include_constants else False,
    )
    # VAE
    compile_vae(
        pipe.vae,
        batch_size=batch_size,
        min_height=min_height,
        max_height=max_height,
        min_width=min_width,
        max_width=max_width,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        constants=True if include_constants else False,
    )


if __name__ == "__main__":
    compile_diffusers()
