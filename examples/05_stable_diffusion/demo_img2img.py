import requests
import torch
from PIL import Image
from io import BytesIO
import click

# from diffusers import StableDiffusionImg2ImgPipeline
from pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline


@click.command()
@click.option("--token", default="", help="access token")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
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
    # or download via git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
    # and pass `model_id_or_path="./stable-diffusion-v1-4"`.
    pipe = pipe.to(device)

    # let's download an initial image
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = "A fantasy landscape, trending on artstation"

    with torch.autocast("cuda"):
        images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images

    images[0].save("fantasy_landscape_ait.png")

if __name__ == "__main__":
    run()
