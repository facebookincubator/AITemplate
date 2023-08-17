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
import click
import numpy as np
import torch

from aitemplate.compiler import Model
from PIL import Image


def esrgan_inference(
    exe_module: Model,
    input_pixels: np.ndarray,
    scale=4,
) -> torch.Tensor:
    if np.max(input_pixels) > 256:
        max_range = 65535
    else:
        max_range = 255
    input_pixels = input_pixels / max_range
    height, width, _ = input_pixels.shape
    inputs = {
        "input_pixels": torch.from_numpy(input_pixels)
        .unsqueeze(0)
        .contiguous()
        .cuda()
        .half(),
    }
    ys = {}
    for name, idx in exe_module.get_output_name_to_index_map().items():
        shape = exe_module.get_output_maximum_shape(idx)
        shape[1] = height * scale
        shape[2] = width * scale
        ys[name] = torch.empty(shape).cuda().half()
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    upscaled = ys["upscaled_pixels"]
    upscaled = upscaled.squeeze(0).cpu().clamp_(0, 1).numpy()
    if max_range == 65535:
        upscaled = (upscaled * 65535.0).round().astype(np.uint16)
    else:
        upscaled = (upscaled * 255.0).round().astype(np.uint8)
    return upscaled


@click.command()
@click.option(
    "--module-path",
    default="./tmp/ESRGANModel/test.so",
    help="the AIT module path",
)
@click.option(
    "--input-image-path",
    default="input.png",
    help="path to input image",
)
@click.option(
    "--output-image-path",
    default="output.png",
    help="path to output image",
)
@click.option(
    "--scale",
    default=4,
    help="Scale of ESRGAN model",
)
def demo(
    module_path,
    input_image_path,
    output_image_path,
    scale,
):
    module = Model(module_path)
    input_image = Image.open(input_image_path).convert("RGB")
    image_array = np.array(input_image)

    upscaled = esrgan_inference(module, image_array, scale)

    output_image = Image.fromarray(upscaled)
    output_image.save(output_image_path)


if __name__ == "__main__":
    demo()
