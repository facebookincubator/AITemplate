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
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option("--batch", default=1, help="Batch size of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative_prompt", default="", help="prompt")
@click.option("--steps", default=50, help="Number of inference steps")
@click.option("--cfg", default=7.5, help="Guidance scale")
def run(
    hf_hub_or_path, ckpt, width, height, batch, prompt, negative_prompt, steps, cfg
):
    pipe = StableDiffusionAITPipeline(
        hf_hub_or_path=hf_hub_or_path,
        ckpt=ckpt,
    )

    prompt = [prompt] * batch
    negative_prompt = [negative_prompt] * batch
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
        ).images[0]
    image.save("example_ait.png")


if __name__ == "__main__":
    run()
