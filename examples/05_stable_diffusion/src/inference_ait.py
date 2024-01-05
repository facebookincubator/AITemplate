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
"""
Helpers for inference using dict[str, torch.Tensor] as inputs and outputs.
input names are manually specified, the same as set in compilation scripts.
output names are taken from the model itself.
usage:
outputs = clip_inference(...)
#Diffusers/Transformers
pooled_prompt_embeds = outputs[0]
prompt_embeds = prompt_embeds.hidden_states[-2]
#AIT
pooled_prompt_embeds = outputs["text_embeds"] # or "pooled_output" is without projection, "text_embeds" is with projection i.e. for bigG
prompt_embeds = outputs["hidden_state_31"]
usage:
latent = unet_inference(...)['latent_output']
usage:
pixels = vae_decode_inference(...)['pixels']
usage:
latent = vae_encode_inference(...)['latent']
"""
from typing import Dict, List

import torch
from aitemplate.compiler import Model


def inference(
    module: Model,
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    benchmark: bool = False,
    benchmark_count: int = 50,
    benchmark_repeat: int = 4,
    permute: bool = False,
    to_cpu: bool = False,
    graph_mode=False,
    sync=True,
):
    module.run_with_tensors(inputs, outputs, graph_mode=graph_mode, sync=sync)
    if permute:
        for name, output in outputs.items():
            if len(output.shape) == 4:
                outputs[name] = output.permute((0, 3, 1, 2))
    if to_cpu:
        for name, output in outputs.items():
            outputs[name] = output.cpu()
    if benchmark:
        t, _, _ = module.benchmark_with_tensors(
            inputs=inputs,
            outputs=outputs,
            count=benchmark_count,
            repeat=benchmark_repeat,
        )
        print(f"latency: {t} ms, it/s: {1000 / t}")

    return outputs


def get_outputs(module: Model, dims, device: str = "cuda", dtype: str = "float16"):
    outputs = {}
    map = module.get_output_name_to_index_map()
    for name, idx in map.items():
        shape = module.get_output_maximum_shape(idx)
        for idx, dim in enumerate(dims):
            shape[idx] = dim
        output = torch.empty(shape).to(device)
        if dtype == "float16":
            output = output.half()
        outputs[name] = output
    return outputs


def timestep_inference(
    module: Model,
    timestep: torch.Tensor,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    to_cpu: bool = False,
    graph_mode: bool = False,
    sync: bool = True
):
    timestep = torch.tensor([timestep]).to(device)
    inputs = {"timestep": timestep.to(device)}
    if dtype == "float16":
        for k, v in inputs.items():
            inputs[k] = v.half()
    dims = [1]
    outputs = get_outputs(module, dims, device, dtype)
    return inference(module, inputs, outputs, benchmark=benchmark, to_cpu=to_cpu, graph_mode=graph_mode, sync=sync)


def clip_inference(
    module: Model,
    input_ids: torch.Tensor,
    seqlen: int = 77,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    to_cpu: bool = False,
    sync: bool = True,
):
    batch = input_ids.shape[0]
    input_ids = input_ids.to(device)
    position_ids = torch.arange(seqlen).expand((batch, -1)).to(device)
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    dims = [batch]
    outputs = get_outputs(module, dims, device, dtype)
    return inference(module, inputs, outputs, benchmark=benchmark, to_cpu=to_cpu, sync=sync)


def unet_inference(
    module: Model,
    latent_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    class_labels: torch.Tensor = None,
    down_block_residuals: List[torch.Tensor] = None,
    mid_block_residual: torch.Tensor = None,
    add_embeds: torch.Tensor = None,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    to_cpu: bool = False,
    graph_mode: bool = False,
    sync: bool = True,
):
    batch = latent_model_input.shape[0]
    height, width = latent_model_input.shape[2], latent_model_input.shape[3]
    timesteps = timesteps.expand(batch)
    inputs = {
        "latent_model_input": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .to(device),
        "timesteps": timesteps.to(device),
        "encoder_hidden_states": encoder_hidden_states.to(device),
    }
    if class_labels is not None:
        inputs["class_labels"] = class_labels.contiguous().to(device)
    if down_block_residuals is not None and mid_block_residual is not None:
        for i, y in enumerate(down_block_residuals):
            inputs[f"down_block_residual_{i}"] = (
                y.permute((0, 2, 3, 1)).contiguous().to(device)
            )
        inputs["mid_block_residual"] = (
            mid_block_residual.permute((0, 2, 3, 1)).contiguous().to(device)
        )
    if add_embeds is not None:
        inputs["add_embeds"] = add_embeds.to(device)
    if dtype == "float16":
        for k, v in inputs.items():
            if k == "class_labels":
                continue
            inputs[k] = v.half()
    dims = [batch, height, width]
    outputs = get_outputs(module, dims, device, dtype)
    return inference(
        module, inputs, outputs, benchmark=benchmark, permute=True, to_cpu=to_cpu, graph_mode=graph_mode, sync=sync,
    )


def vae_decode_inference(
    module: Model,
    latent: torch.Tensor,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    factor: int = 8,
    to_cpu: bool = False,
    graph_mode=False,
    sync: bool = True,
):
    batch = latent.shape[0]
    height, width = latent.shape[2:]
    height *= factor
    width *= factor
    latent = latent.permute((0, 2, 3, 1)).contiguous().to(device)
    if dtype == "float16":
        latent = latent.half()
    inputs = {
        "latent": latent,
    }
    dims = [batch, height, width]
    outputs = get_outputs(module, dims, device, dtype)
    return inference(
        module, inputs, outputs, benchmark=benchmark, permute=True, to_cpu=to_cpu, graph_mode=graph_mode, sync=sync
    )


def vae_encode_inference(
    module: Model,
    pixels: torch.Tensor,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    factor: int = 8,
    latent_channels: int = 4,
    to_cpu: bool = False,
):
    batch = pixels.shape[0]
    height, width = pixels.shape[2:]
    height *= factor
    width *= factor
    pixels = pixels.permute((0, 2, 3, 1)).contiguous().to(device)
    sample = torch.randn(batch, height, width, latent_channels).to(device)
    if dtype == "float16":
        pixels = pixels.half()
        sample = sample.half()
    inputs = {
        "pixels": pixels,
        "random_sample": sample,
    }
    dims = [batch, height, width]
    outputs = get_outputs(module, dims, device, dtype)
    return inference(
        module, inputs, outputs, benchmark=benchmark, permute=True, to_cpu=to_cpu
    )
