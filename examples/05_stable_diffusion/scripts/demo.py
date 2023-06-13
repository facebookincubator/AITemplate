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

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from diffusers import EulerDiscreteScheduler

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="the local diffusers pipeline directory",
)
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option("--batch", default=1, help="Batch size of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative_prompt", default="", help="prompt")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(local_dir, width, height, batch, prompt, negative_prompt, benchmark):
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    prompt = [prompt] * batch
    with torch.autocast("cuda"):
        images = pipe(prompt, height, width).images
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(
                f"sd e2e: width={width}, height={height}, batchsize={batch}, latency={t} ms"
            )
    for i, image in enumerate(images):
        image.save(f"example_ait_{i}.png")


if __name__ == "__main__":
    run()
