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
import os

import click
import torch
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from timm.models.vision_transformer import VisionTransformer
from torch import nn


def create_vit(model_name):
    if model_name == "vit_base_patch16_224":
        img_size = 224
        embed_dim = 768
        class_token = False
        global_pool = "avg"
        depth = 12
        patch_size = 16
        num_heads = 12
    elif model_name == "vit_large_patch16_384":
        img_size = 384
        embed_dim = 1024
        class_token = False
        global_pool = "avg"
        depth = 24
        patch_size = 16
        num_heads = 16
    else:
        raise NotImplementedError
    model = (
        VisionTransformer(
            img_size=img_size,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            class_token=class_token,
            global_pool=global_pool,
            depth=depth,
            patch_size=patch_size,
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        .cuda()
        .half()
    )
    return model


def benchmark(model_name, batch_size, img_size):
    if model_name == "vit_base_patch16_224":
        img_size = 224
    elif model_name == "vit_large_patch16_384":
        img_size = 384
    model = create_vit(model_name)
    with torch.inference_mode():
        input_shape = (batch_size, 3, img_size, img_size)
        input_data = torch.randn(input_shape).cuda().half()
        # warm up
        benchmark_torch_function(100, model, input_data)
        # benchmark
        t = benchmark_torch_function(100, model, input_data)
        print("batch_size: {}, time: {}".format(batch_size, t))
        dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
        dev_flag = dev_flag.replace(",", "_")
        with open(f"{model_name}_pt_benchmark_dev_{dev_flag}.txt", "a") as f:
            f.write("batch_size: {}, latency: {}\n".format(batch_size, t))


@click.command()
@click.option("--model-name", type=str, default="vit_base_patch16_224")
@click.option("--batch-size", default=0, type=int)
def main(model_name, batch_size):
    img_size = 224
    if model_name == "vit_base_patch16_224":
        img_size = 224
    elif model_name == "vit_large_patch16_384":
        img_size = 384
    else:
        raise NotImplementedError
    if batch_size == 0:
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            benchmark(model_name, batch_size, img_size)
    else:
        benchmark(model_name, batch_size, img_size)


if __name__ == "__main__":
    main()
