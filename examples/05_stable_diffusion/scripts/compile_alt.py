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
@click.option(
    "--width",
    default=(64, 2048),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum width",
)
@click.option(
    "--height",
    default=(64, 2048),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum height",
)
@click.option(
    "--batch-size",
    default=(1, 4),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option("--clip-chunks", default=6, help="Maximum number of clip chunks")
@click.option(
    "--include-constants",
    default=None,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--controlnet", default=False, help="UNet for controlnet")
def compile_diffusers(
    local_dir,
    width,
    height,
    batch_size,
    clip_chunks,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    controlnet=False,
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    assert (
        width[0] % 64 == 0 and width[1] % 64 == 0
    ), "Minimum Width and Maximum Width must be multiples of 64, otherwise, the compilation process will fail."
    assert (
        height[0] % 64 == 0 and height[1] % 64 == 0
    ), "Minimum Height and Maximum Height must be multiples of 64, otherwise, the compilation process will fail."

    pipe = StableDiffusionPipeline.from_pretrained(
        local_dir,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

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
        batch_size=batch_size,
        width=width,
        height=height,
        clip_chunks=clip_chunks,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        hidden_dim=pipe.unet.config.cross_attention_dim,
        attention_head_dim=pipe.unet.config.attention_head_dim,
        use_linear_projection=pipe.unet.config.get("use_linear_projection", False),
        constants=True if include_constants else False,
        controlnet=True if controlnet else False,
    )
    # VAE
    compile_vae(
        pipe.vae,
        batch_size=batch_size,
        width=width,
        height=height,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        constants=True if include_constants else False,
    )


if __name__ == "__main__":
    compile_diffusers()
