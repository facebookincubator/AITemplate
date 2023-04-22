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

import logging

import click

import numpy as np
import torch
from aitemplate.compiler import Model
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from diffusers import StableDiffusionPipeline

from torch import autocast
from transformers import CLIPTokenizer

USE_CUDA = detect_target().name() == "cuda"


def get_int_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))


def benchmark_unet(
    pt_mod,
    batch_size=2,
    height=64,
    width=64,
    dim=320,
    hidden_dim=1024,
    benchmark_pt=False,
    verify=False,
):

    exe_module = Model("./tmp/UNet2DConditionModel/test.so")
    if exe_module is None:
        print("Error!! Cannot find compiled module for UNet2DConditionModel.")
        exit(-1)

    # run PT unet model
    pt_mod = pt_mod.eval()

    latent_model_input_pt = torch.randn(batch_size, 4, height, width).cuda().half()
    text_embeddings_pt = torch.randn(batch_size, 64, hidden_dim).cuda().half()
    timesteps_pt = torch.Tensor([1, 1]).cuda().half()

    with autocast("cuda"):
        pt_ys = pt_mod(
            latent_model_input_pt,
            timesteps_pt,
            encoder_hidden_states=text_embeddings_pt,
        ).sample

        # PT benchmark
        if benchmark_pt:
            args = (latent_model_input_pt, 1, text_embeddings_pt)
            pt_time = benchmark_torch_function(100, pt_mod, *args)
            print(f"PT batch_size: {batch_size}, {pt_time} ms")
            with open("sd_pt_benchmark.txt", "a") as f:
                f.write(f"unet batch_size: {batch_size}, latency: {pt_time} ms\n")

    print("pt output:", pt_ys.shape)

    # run AIT unet model
    inputs = {
        "input0": latent_model_input_pt.permute((0, 2, 3, 1)).contiguous(),
        "input1": timesteps_pt,
        "input2": text_embeddings_pt,
    }

    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys)

    # verification
    y_transpose = ys[0].permute((0, 3, 1, 2))

    if verify:
        eps = 1e-1
        np.testing.assert_allclose(
            pt_ys.detach().cpu().numpy(),
            y_transpose.cpu().numpy(),
            atol=eps,
            rtol=eps,
        )
        print("UNet2DCondition verification pass")

    # AIT benchmark
    # warmup
    exe_module.benchmark_with_tensors(inputs, ys, count=100, repeat=4)
    # benchmark
    t, _, _ = exe_module.benchmark_with_tensors(inputs, ys, count=100, repeat=4)
    with open("sd_ait_benchmark.txt", "a") as f:
        f.write(f"unet batch_size: {batch_size}, latency: {t} ms\n")


def benchmark_clip(
    pt_mod,
    batch_size=1,
    seqlen=64,
    tokenizer=None,
    benchmark_pt=False,
    verify=False,
):
    mask_seq = 0

    exe_module = Model("./tmp/CLIPTextModel/test.so")
    if exe_module is None:
        print("Error!! Cannot find compiled module for CLIPTextModel.")
        exit(-1)

    # run PT clip
    pt_mod = pt_mod.eval()

    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_input = tokenizer(
        ["a photo of an astronaut riding a horse on mars"],
        padding="max_length",
        max_length=seqlen,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input["input_ids"].cuda()

    attention_mask = torch.ones((batch_size, seqlen))
    attention_mask[-1, -mask_seq:] = 0
    attention_mask = None

    position_ids = torch.arange(seqlen).expand((batch_size, -1)).cuda()
    pt_ys = pt_mod(input_ids, attention_mask, position_ids)
    print("pt output:", pt_ys[0].shape)

    # PT benchmark
    if benchmark_pt:
        args = (input_ids, attention_mask, position_ids)
        pt_time = benchmark_torch_function(100, pt_mod, *args)
        print(f"PT batch_size: {batch_size}, {pt_time} ms")
        with open("sd_pt_benchmark.txt", "a") as f:
            f.write(f"clip batch_size: {batch_size}, latency: {pt_time} ms\n")

    # run AIT clip
    inputs = {
        "input0": input_ids,
        "input1": position_ids,
    }
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys)

    # verification
    if verify:
        eps = 1e-1
        pt_np = pt_ys[0].detach().cpu().numpy()
        np.testing.assert_allclose(
            pt_np,
            ys[0].cpu().numpy(),
            atol=eps,
            rtol=eps,
        )
        print("CLIPTextTransformer verification pass")

    # AIT benchmark
    # warmup
    exe_module.benchmark_with_tensors(inputs, ys, count=100, repeat=4)
    # benchmark
    t, _, _ = exe_module.benchmark_with_tensors(inputs, ys, count=100, repeat=4)
    with open("sd_ait_benchmark.txt", "a") as f:
        f.write(f"clip batch_size: {batch_size}, latency: {t} ms\n")


def benchmark_vae(
    pt_vae, batch_size=1, height=64, width=64, benchmark_pt=False, verify=False
):

    latent_channels = 4

    exe_module = Model("./tmp/AutoencoderKL/test.so")
    if exe_module is None:
        print("Error!! Cannot find compiled module for AutoencoderKL.")
        exit(-1)

    # run PT vae
    pt_vae = pt_vae.cuda().half()
    pt_vae.eval()

    pt_input = torch.rand([batch_size, latent_channels, height, width]).cuda().half()
    print("pt_input shape", pt_input.shape)
    with autocast("cuda"):
        pt_output = pt_vae.decode(pt_input).sample
        pt_output = pt_output.half()

        # PT benchmark
        if benchmark_pt:
            args = (pt_input,)
            pt_time = benchmark_torch_function(100, pt_vae.decode, *args)
            print(f"PT batch_size: {batch_size}, {pt_time} ms")
            with open("sd_pt_benchmark.txt", "a") as f:
                f.write(f"vae batch_size: {batch_size}, latency: {pt_time} ms\n")

    # run AIT vae
    y = (
        torch.empty(
            pt_output.size(0),
            pt_output.size(2),
            pt_output.size(3),
            pt_output.size(1),
        )
        .cuda()
        .half()
    )
    ait_input_pt_tensor = torch.permute(pt_input, (0, 2, 3, 1)).contiguous()
    print("input pt tensor size: ", ait_input_pt_tensor.shape)
    print("output pt tensor size: ", y.shape)
    exe_module.run_with_tensors([ait_input_pt_tensor], [y])

    # verification
    if verify:
        y_pt = torch.permute(y, (0, 3, 1, 2))
        eps = 1e-1
        np.testing.assert_allclose(
            pt_output.detach().cpu().numpy(),
            y_pt.cpu().numpy(),
            atol=eps,
            rtol=eps,
        )
        logging.info("VAE Verification done!")

    # AIT benchmark:
    # warmup
    exe_module.benchmark_with_tensors([ait_input_pt_tensor], [y], count=100, repeat=4)
    # benchmark
    t, _, _ = exe_module.benchmark_with_tensors(
        [ait_input_pt_tensor], [y], count=100, repeat=4
    )
    with open("sd_ait_benchmark.txt", "a") as f:
        f.write(f"vae batch_size: {batch_size}, latency: {t} ms\n")


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="the local diffusers pipeline directory",
)
@click.option("--batch-size", default=1, help="batch size")
@click.option("--verify", type=bool, default=False, help="verify correctness")
@click.option("--benchmark-pt", type=bool, default=False, help="run pt benchmark")
def benchmark_diffusers(local_dir, batch_size, verify, benchmark_pt):
    assert batch_size == 1, "batch size must be 1 for submodule verification"
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    pipe = StableDiffusionPipeline.from_pretrained(
        local_dir,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    # CLIP
    benchmark_clip(
        pipe.text_encoder,
        batch_size=batch_size,
        benchmark_pt=benchmark_pt,
        verify=verify,
    )
    # UNet
    benchmark_unet(
        pipe.unet,
        batch_size=batch_size * 2,
        benchmark_pt=benchmark_pt,
        verify=verify,
        hidden_dim=pipe.text_encoder.config.hidden_size,
    )
    # VAE
    benchmark_vae(
        pipe.vae, batch_size=batch_size, benchmark_pt=benchmark_pt, verify=verify
    )


if __name__ == "__main__":
    benchmark_diffusers()
