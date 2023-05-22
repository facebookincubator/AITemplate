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
from diffusers import ControlNetModel

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.compile_lib.compile_controlnet import compile_controlnet


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/runwayml/stable-diffusion-v1-5",
    help="the local diffusers pipeline directory",
)
@click.option("--width", default=512, type=int, help="width")
@click.option("--height", default=512, type=int, help="height")
@click.option("--batch-size", default=1, type=int, help="batch size")
@click.option("--clip-chunks", default=6, help="Maximum number of clip chunks")
@click.option(
    "--include-constants",
    default=None,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def compile_diffusers(
    local_dir,
    width,
    height,
    batch_size,
    clip_chunks,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    assert (
        width % 64 == 0
    ), "Width must be multiples of 64, otherwise, the compilation process will fail."
    assert (
        height % 64 == 0
    ), "Height must be multiples of 64, otherwise, the compilation process will fail."

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    ).to("cuda")

    compile_controlnet(
        controlnet,
        batch_size=batch_size,
        width=width,
        height=height,
        clip_chunks=clip_chunks,
        convert_conv_to_gemm=convert_conv_to_gemm,
        use_fp16_acc=use_fp16_acc,
        constants=include_constants,
    )


if __name__ == "__main__":
    compile_diffusers()
