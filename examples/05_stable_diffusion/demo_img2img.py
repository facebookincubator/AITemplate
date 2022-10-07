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
import requests
import torch
from PIL import Image

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline


@click.command()
@click.option("--token", default="", help="access token")
@click.option(
    "--prompt", default="A fantasy landscape, trending on artstation", help="prompt"
)
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(token, prompt, benchmark):

    # load the pipeline
    device = "cuda"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgAITPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=token,
    )
    pipe = pipe.to(device)

    # let's download an initial image
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))

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
