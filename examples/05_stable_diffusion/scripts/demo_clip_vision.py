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
import torch

from aitemplate.compiler import Model
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


def clip_vision_inference(
    exe_module: Model,
    pixel_values: torch.Tensor,
    max_positions: int = 257,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
):
    print(pixel_values.shape)
    batch = pixel_values.shape[0]
    pixel_values = pixel_values.to(device)
    position_ids = torch.arange(max_positions).expand((batch, -1)).to(device)
    inputs = {
        "pixel_values": pixel_values.permute((0, 2, 3, 1))
        .contiguous()
        .to(device)
        .half(),
        "position_ids": position_ids,
    }
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        print(shape)
        shape[0] = batch
        ys.append(torch.empty(shape).to(device))
        if dtype == "float16":
            ys[i] = ys[i].half()
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    if benchmark:
        # warm up
        _, _, _ = exe_module.benchmark_with_tensors(
            inputs=inputs,
            outputs=ys,
            count=50,
            repeat=4,
        )
        t, _, _ = exe_module.benchmark_with_tensors(
            inputs=inputs,
            outputs=ys,
            count=50,
            repeat=4,
        )
        print(f"AIT clip_vision latency: {t} ms, it/s: {1000 / t}")
    return ys


@click.command()
@click.option(
    "--module-path",
    default="./tmp/CLIPVisionModel/test.so",
    help="Path to the compiled module",
)
@click.option("--image-path", help="Path to the input image")
@click.option(
    "--image-embeds", type=bool, default=False, help="Whether to return image embeds"
)
@click.option(
    "--hf-hub-or-path",
    help="Path to PyTorch model",
    default="openai/clip-vit-large-patch14",
)
@click.option(
    "--benckmark",
    type=bool,
    default=False,
    help="Whether to benchmark",
)
def run(
    module_path,
    image_path,
    image_embeds=False,
    hf_hub_or_path="openai/clip-vit-large-patch14",
    benchmark=False,
):
    if image_embeds:
        pt_mod = CLIPVisionModelWithProjection.from_pretrained(
            hf_hub_or_path, torch_dtype=torch.float16
        ).cuda()
    else:
        pt_mod = CLIPVisionModel.from_pretrained(
            hf_hub_or_path, torch_dtype=torch.float16
        ).cuda()

    ait_mod = Model(module_path)

    image = Image.open(image_path).convert("RGB")

    processor = CLIPImageProcessor.from_pretrained(hf_hub_or_path)

    pixel_values = processor(image, return_tensors="pt").pixel_values

    ait_out = clip_vision_inference(ait_mod, pixel_values, benchmark)
    print("AIT")
    for tensor in ait_out:
        if image_embeds:
            tensor /= tensor.norm(dim=-1, keepdim=True)
        print(tensor.shape)
        print(tensor.cpu().numpy())

    pixel_values = pixel_values.cuda().half()
    pt_out = pt_mod(pixel_values)
    print("PT")
    for tensor in pt_out:
        if image_embeds:
            tensor /= tensor.norm(dim=-1, keepdim=True)
        print(tensor.shape)
        print(tensor.cpu().numpy())
    if benchmark:
        t = benchmark_torch_function(50, pt_mod, pixel_values)
        print(f"PT clip_vision latency: {t} ms, it/s: {1000 / t}")
