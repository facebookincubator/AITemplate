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
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.compile_lib.compile_clip_vision import compile_clip_vision


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="openai/clip-vit-large-patch14",
    help="the local diffusers pipeline directory or hf hub path e.g. openai/clip-vit-large-patch14",
)
@click.option(
    "--batch-size",
    default=(1, 2),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option(
    "--image-embeds",
    default=False,
    type=bool,
    help="Whether to return image embeddings",
)
@click.option(
    "--include-constants",
    default=False,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--model-name", default="CLIPVisionModel", help="module name")
@click.option("--work-dir", default="A:/", help="work directory")
def compile_diffusers(
    hf_hub_or_path,
    batch_size,
    image_embeds,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    model_name="CLIPVisionModel",
    work_dir="./tmp",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    if image_embeds:
        pipe = CLIPVisionModelWithProjection.from_pretrained(
            hf_hub_or_path,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        pipe = CLIPVisionModel.from_pretrained(
            hf_hub_or_path,
            torch_dtype=torch.float16,
        ).to("cuda")

    compile_clip_vision(
        pipe,
        batch_size=batch_size,
        hidden_size=pipe.config.hidden_size,
        projection_dim=pipe.config.projection_dim if image_embeds else None,
        num_hidden_layers=pipe.config.num_hidden_layers,
        num_attention_heads=pipe.config.num_attention_heads,
        patch_size=pipe.config.patch_size,
        image_size=pipe.config.image_size,
        layer_norm_eps=pipe.config.layer_norm_eps,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        act_layer=pipe.config.hidden_act,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    compile_diffusers()
