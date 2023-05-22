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
import cv2
import numpy as np
import torch
from aitemplate.utils.import_path import import_parent
from diffusers.utils import load_image
from PIL import Image

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_controlnet_ait import StableDiffusionAITPipeline


def prepare_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image[0], Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=Image.LANCZOS)
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image


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
    # download an image
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )
    image = np.array(image)
    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    control_cond = prepare_image(
        canny_image,
        width,
        height,
        batch,
        1,
        "cuda",
        torch.float16,
        do_classifier_free_guidance=True,
    )
    prompt = [prompt] * batch
    negative_prompt = [negative_prompt] * batch
    with torch.autocast("cuda"):
        for _ in range(5):
            image = pipe(
                prompt=prompt,
                control_cond=control_cond,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
            ).images[0]
    image.save("example_ait.png")


if __name__ == "__main__":
    run()
