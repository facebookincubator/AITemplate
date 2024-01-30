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
from diffusers import DiffusionPipeline


@click.command()
@click.option(
    "--model-name",
    default="stabilityai/stable-diffusion-2-1-base",
    help="Pretrained Model name.",
)
@click.option(
    "--token",
    default="",
    help="Valid values: Huggingface user access token, 'true' to use token "
    "generated with 'huggingface-cli login' (stored in ~/.huggingface) "
    "or empty string to not use access token (default).",
)
@click.option(
    "--save-directory",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="Pipeline files local directory.",
)
def download_pipeline_files(model_name, token, save_directory) -> None:

    DiffusionPipeline.from_pretrained(
        model_name,
        revision="main" if "xl" in model_name else "fp16",
        torch_dtype=torch.float16,
        use_auth_token=token if len(token) > 5 else token.lower() == "true",
    ).save_pretrained(save_directory)


if __name__ == "__main__":
    download_pipeline_files()