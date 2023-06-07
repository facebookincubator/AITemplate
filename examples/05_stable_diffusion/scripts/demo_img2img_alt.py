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
from io import BytesIO

import click
import torch
import requests

from PIL import Image
from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait_alt import StableDiffusionAITPipeline


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="runwayml/stable-diffusion-v1-5",
    help="Model weights to apply to compiled model (with --include-constants false)",
)
@click.option("--ckpt", default=None, help="e.g. v1-5-pruned-emaonly.ckpt")
@click.option("--width", default=768, help="Width of generated image")
@click.option("--height", default=768, help="Height of generated image")
@click.option("--batch", default=1, help="Batch size of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative_prompt", default="", help="prompt")
@click.option("--steps", default=30, help="Number of inference steps")
@click.option("--cfg", default=7.5, help="Guidance scale")
@click.option("--strength", default=0.5, help="Guidance scale")
@click.option("--workdir", default="v15", help="Workdir")
def run(
        hf_hub_or_path, ckpt, width, height, batch, prompt, negative_prompt, steps, cfg, strength, workdir
):
    pipe = StableDiffusionAITPipeline(
        workdir=workdir,
        hf_hub_or_path=hf_hub_or_path,
        ckpt=ckpt,
    )

    prompt = [prompt] * batch
    negative_prompt = [negative_prompt] * batch

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((height, width))

    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            init_image=init_image,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=strength,
        ).images[0]
    image.save("example_ait.png")


if __name__ == "__main__":
    run()
