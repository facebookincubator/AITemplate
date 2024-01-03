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
from diffusers import AutoencoderKL, DiffusionPipeline

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_xl_ait import StableDiffusionXLAITPipeline
from aitemplate.testing.benchmark_pt import benchmark_torch_function


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="stabilityai/stable-diffusion-xl-base-1.0",
    help="huggingface hub name or path to local model",
)
@click.option(
    "--apply-weights",
    default=True,
    help="apply weights to module, required for Windows",
)
@click.option(
    "--unet-module",
    help="path to unet module",
    required=True,
)
@click.option(
    "--text-encoder-module",
    help="path to text encoder module",
    required=True,
)
@click.option(
    "--text-encoder-2-module",
    help="path to text encoder 2 module",
    required=True,
)
@click.option(
    "--time-embed-module",
    help="path to time embed module",
    required=True,
)
@click.option(
    "--vae-module",
    help="path to vae module",
    required=True,
)
@click.option("--width", default=1024, help="Width of generated image")
@click.option("--height", default=1024, help="Height of generated image")
@click.option("--batch", default=1, help="Batch size of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative_prompt", default="", help="prompt")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(
    hf_hub_or_path,
    apply_weights,
    unet_module,
    text_encoder_module,
    text_encoder_2_module,
    time_embed_module,
    vae_module,
    width,
    height,
    batch,
    prompt,
    negative_prompt,
    benchmark,
):
    diffusers_pipe = DiffusionPipeline.from_pretrained(
        hf_hub_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe = StableDiffusionXLAITPipeline(
        vae,
        diffusers_pipe.text_encoder,
        diffusers_pipe.text_encoder_2,
        diffusers_pipe.tokenizer,
        diffusers_pipe.tokenizer_2,
        diffusers_pipe.unet,
        diffusers_pipe.scheduler,
        text_encoder_module,
        text_encoder_2_module,
        unet_module,
        vae_module,
        time_embed_module,
        apply_weights_to_modules=apply_weights,
    )

    prompt = [prompt] * batch
    images = pipe(
        prompt=prompt,
        prompt_2=prompt,
        height=height,
        width=width,
        num_inference_steps=20,
        guidance_scale=8
    ).images
    
    for i, image in enumerate(images):
        image.save(f"example_ait_{i}.png")

    if benchmark:
        t = benchmark_torch_function(10, pipe, prompt, prompt_2=prompt, height=height, width=width, num_inference_steps=20, guidance_scale=8)
        print(
            f"sd e2e: width={width}, height={height}, batchsize={batch}, latency={t} ms"
        )



if __name__ == "__main__":
    run()
