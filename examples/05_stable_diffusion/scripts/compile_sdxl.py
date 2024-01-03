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
import sys

sys.setrecursionlimit(10000)

import click
import torch
from aitemplate.testing import detect_target
from aitemplate.utils.import_path import import_parent
from diffusers import AutoencoderKL, DiffusionPipeline

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.compile_lib.compile_clip_alt import compile_clip
from src.compile_lib.compile_unet_alt import compile_timestep_embedder, compile_unet
from src.compile_lib.compile_vae_alt import compile_vae


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="stabilityai/stable-diffusion-xl-base-1.0",
    help="the local or hf hub path e.g. stabilityai/stable-diffusion-xl-base-1.0",
)
@click.option(
    "--width",
    default=(1024, 1024),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum width",
)
@click.option(
    "--height",
    default=(1024, 1024),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum height",
)
@click.option(
    "--batch-size",
    default=(1, 1),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option("--clip-chunks", default=10, help="Maximum number of clip chunks")
@click.option(
    "--include-constants",
    default=False,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--work-dir", default="./tmp", help="work directory")
@click.option(
    "--model-name-prefix", default="SDXL", help="Prefix for compiled module names"
)
@click.option(
    "--fp32-vae",
    default=False,
    help="fp32 vae, if false, use https://huggingface.co/madebyollin/sdxl-vae-fp16-fix as replacement vae",
)
def compile_diffusers(
    hf_hub_or_path,
    width,
    height,
    batch_size,
    clip_chunks,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    work_dir="./tmp",
    model_name_prefix="SDXL",
    fp32_vae=False,
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = DiffusionPipeline.from_pretrained(
        hf_hub_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    if fp32_vae:
        pipe.vae.to("cuda", dtype=torch.float32)
    else:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to("cuda")
        pipe.vae = vae

    # text_encoder
    model_name = f"{model_name_prefix}_text_encoder"
    compile_clip(
        pipe.text_encoder,
        batch_size=batch_size,
        seqlen=pipe.text_encoder.config.max_position_embeddings,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        output_hidden_states=True,
        text_projection_dim=None,
        depth=pipe.text_encoder.config.num_hidden_layers,
        num_heads=pipe.text_encoder.config.num_attention_heads,
        dim=pipe.text_encoder.config.hidden_size,
        act_layer=pipe.text_encoder.config.hidden_act,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
    )
    # text_encoder 2
    model_name = f"{model_name_prefix}_text_encoder_2"
    compile_clip(
        pipe.text_encoder_2,
        batch_size=batch_size,
        seqlen=pipe.text_encoder_2.config.max_position_embeddings,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        output_hidden_states=True,
        text_projection_dim=pipe.text_encoder_2.config.projection_dim,
        depth=pipe.text_encoder_2.config.num_hidden_layers,
        num_heads=pipe.text_encoder_2.config.num_attention_heads,
        dim=pipe.text_encoder_2.config.hidden_size,
        act_layer=pipe.text_encoder_2.config.hidden_act,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
    )
    model_name = f"{model_name_prefix}_unet"
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
        block_out_channels=pipe.unet.config.block_out_channels,
        down_block_types=pipe.unet.config.down_block_types,
        up_block_types=pipe.unet.config.up_block_types,
        in_channels=pipe.unet.config.in_channels,
        out_channels=pipe.unet.config.out_channels,
        class_embed_type=pipe.unet.config.class_embed_type,
        num_class_embeds=pipe.unet.config.num_class_embeds,
        only_cross_attention=pipe.unet.config.only_cross_attention,
        sample_size=pipe.unet.config.sample_size,
        dim=pipe.unet.config.block_out_channels[0],
        time_embedding_dim=pipe.unet.config.time_embedding_dim,
        conv_in_kernel=pipe.unet.config.conv_in_kernel,
        projection_class_embeddings_input_dim=pipe.unet.config.projection_class_embeddings_input_dim,
        addition_embed_type=pipe.unet.config.addition_embed_type,
        transformer_layers_per_block=pipe.unet.config.transformer_layers_per_block,
        constants=False
        if sys.platform == "win32"
        else include_constants,  # Too big, RC : fatal error RW1023: I/O error seeking in file
        model_name=model_name,
        work_dir=work_dir,
    )
    # `add_time_proj` Timesteps
    model_name = f"{model_name_prefix}_addition_time_embed"
    compile_timestep_embedder(
        pipe.unet.config.addition_time_embed_dim,
        work_dir=work_dir,
        model_name=model_name,
    )
    model_name = f"{model_name_prefix}_vae"
    # VAE
    compile_vae(
        pipe.vae,
        batch_size=batch_size,
        width=width,
        height=height,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        constants=include_constants,
        block_out_channels=pipe.vae.config.block_out_channels,
        layers_per_block=pipe.vae.config.layers_per_block,
        act_fn=pipe.vae.config.act_fn,
        latent_channels=pipe.vae.config.latent_channels,
        in_channels=pipe.vae.config.in_channels,
        out_channels=pipe.vae.config.out_channels,
        down_block_types=pipe.vae.config.down_block_types,
        up_block_types=pipe.vae.config.up_block_types,
        sample_size=pipe.vae.config.sample_size,
        model_name=model_name,
        work_dir=work_dir,
        dtype="float32" if fp32_vae else "float16",
    )


if __name__ == "__main__":
    compile_diffusers()
