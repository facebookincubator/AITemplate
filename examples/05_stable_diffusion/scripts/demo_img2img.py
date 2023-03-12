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
import os
import click
import requests
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from PIL import Image

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="the local diffusers pipeline directory",
)
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option(
    "--prompt", default="A fantasy landscape, trending on artstation", help="prompt"
)
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(local_dir, width, height, prompt, benchmark):

     
    """
        Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute path to a directory which 
        will be used to save the AIT compiled model artifacts. Make sure the OS user running this script has read and write 
        permissions to this directory. By default, the artifacts will be saved under tmp/ folder of the 
        current working directory. 
    """

    env_name = "AITEMPLATE_WORK_DIR"
    try:
        if os.environ[env_name]:
            local_dir = os.path.join(os.environ[env_name], 'diffusers-pipeline', 'stabilityai','stable-diffusion-v2')
            print("The value of", env_name, " is ", local_dir)
    except KeyError:
        print("AITEMPLATE_WORK_DIR environment variable is not set. Using default local dir as ",local_dir)

    # load the pipeline
    device = "cuda"
    pipe = StableDiffusionImg2ImgAITPipeline.from_pretrained(
        local_dir,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = pipe.to(device)
    # let's download an initial image
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((height, width))

    with torch.autocast("cuda"):
        images = pipe(
            prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5
        ).images
        if benchmark:
            args = (prompt, init_image)
            t = benchmark_torch_function(10, pipe, *args)
            print(f"sd e2e: {t} ms")

    images[0].save("fantasy_landscape_ait.png")


if __name__ == "__main__":
    run()
