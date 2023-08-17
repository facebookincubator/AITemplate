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
import safetensors.torch
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from modeling.rrdbnet import map_rrdb, mark_output, RRDBNet


@click.command()
@click.option(
    "--model-path",
    default="RealESRGAN_x4plus.safetensors",
    help="model path. supports torch or safetensors",
)
@click.option(
    "--width",
    default=(64, 1024),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum width",
)
@click.option(
    "--height",
    default=(64, 1024),
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
@click.option(
    "--include-constants",
    default=True,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option(
    "--num-in-ch",
    default=3,
    type=int,
    help="Number of in channels",
)
@click.option(
    "--num-out-ch",
    default=3,
    type=int,
    help="Number of out channels",
)
@click.option(
    "--num-feat",
    default=64,
    type=int,
    help="Number of intermediate features",
)
@click.option(
    "--num-block",
    default=23,
    type=int,
    help="Number of RRDB layers",
)
@click.option(
    "--num-grow-ch",
    default=32,
    type=int,
    help="Number of channels for each growth",
)
@click.option(
    "--scale",
    default=4,
    type=int,
    help="Scale",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--work-dir", default="./tmp", help="Work directory")
@click.option("--model-name", default="ESRGANModel", help="Model name")
def compile_esrgan(
    model_path,
    width,
    height,
    batch_size,
    include_constants,
    num_in_ch,
    num_out_ch,
    num_feat,
    num_block,
    num_grow_ch,
    scale,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    work_dir="./tmp",
    model_name="ESRGANModel",
):
    if scale != 4:
        print(
            "Scale != 4 supports static shape only. Maximum value of batch_size, height and width will be used."
        )

    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    if model_path.endswith(".safetensors"):
        pt_model = safetensors.torch.load_file(model_path)
    else:
        pt_model = torch.load(model_path)

    if "params_ema" in pt_model.keys():
        pt_model = pt_model["params_ema"]
    elif "params" in pt_model.keys():
        pt_model = pt_model["params"]

    rrdbnet = RRDBNet(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
    )
    rrdbnet.name_parameter_tensor()

    constants = map_rrdb(pt_model, scale=scale)

    batch_size = IntVar(values=list(batch_size), name="batch_size")
    channels = num_in_ch
    height = IntVar(values=list(height), name="height")
    width = IntVar(values=list(width), name="width")

    image = Tensor(
        shape=[batch_size, height, width, channels], name="input_pixels", is_input=True
    )

    Y = rrdbnet(image)
    Y = mark_output(Y, "upscaled_pixels")

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        work_dir,
        model_name,
        constants=constants if include_constants else None,
    )


if __name__ == "__main__":
    compile_esrgan()
