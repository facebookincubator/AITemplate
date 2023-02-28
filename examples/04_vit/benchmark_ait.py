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
"""benchmark for vit"""

import os

import click
import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from modeling.vision_transformer import VisionTransformer
from weight_utils import export_to_torch_tensor

# flake8: noqa


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


USE_CUDA = detect_target().name() == "cuda"


def compile_vit(
    model_name,
    batch_size,
    class_token=False,
    global_pool="avg",
    use_fp16_acc=True,
):
    img_size = 224
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    depth = 12
    if model_name == "vit_base_patch16_224":
        img_size = 224
        patch_size = 16
        embed_dim = 768
        num_heads = 12
        depth = 12
    elif model_name == "vit_large_patch16_384":
        img_size = 384
        patch_size = 16
        embed_dim = 1024
        num_heads = 16
        depth = 24
    seqlen = (img_size // patch_size) ** 2 + (1 if class_token else 0)
    ait_model = VisionTransformer(
        batch_size=batch_size,
        img_size=img_size,
        class_token=class_token,
        global_pool=global_pool,
        num_heads=num_heads,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        act_layer="GELU",
    )
    ait_model.name_parameter_tensor()
    inputs_ait = Tensor(
        [batch_size, img_size, img_size, 3], name="input0", is_input=True
    )
    Y = ait_model(inputs_ait)
    mark_output(Y)

    target = detect_target(use_fp16_acc=use_fp16_acc)
    exe_module = compile_model(
        Y, target, "./tmp", "vision_transformer_bs%d_seq%d" % (batch_size, seqlen)
    )
    return exe_module


def benchmark(model_name, batch_size, mod=None, graph_mode=True):
    # load mod
    if model_name == "vit_base_patch16_224":
        img_size = 224
        patch_size = 16
        embed_dim = 768
        num_heads = 12
        depth = 12
    elif model_name == "vit_large_patch16_384":
        img_size = 384
        patch_size = 16
        embed_dim = 1024
        num_heads = 16
        depth = 24
    else:
        raise NotImplementedError

    seqlen = (img_size // patch_size) ** 2

    if mod is None:
        model_dir = f"vision_transformer_bs{batch_size}_seq{seqlen}"
        mod = Model(os.path.join("./tmp", model_dir, "test.so"))

    # prepare params
    params_ait = export_to_torch_tensor(model_name)
    if detect_target().name() == "cuda":
        ait_key = "attn_cu_length"
        for i in range(depth):
            prefix = "blocks_%d" % (i)
            cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
            params_ait[f"{prefix}_{ait_key}"] = torch.from_numpy(cu_len).cuda()

    # set weights
    mod.set_many_constants_with_tensors(params_ait)
    mod.fold_constants(sync=True)

    # prepare input/output tensor
    inputs = [torch.randn([batch_size, img_size, img_size, 3]).cuda().half()]
    ys = []
    num_outputs = len(mod.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = mod.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    # warm up
    t, _, __ = mod.benchmark_with_tensors(
        inputs,
        ys,
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    # benchmark
    t, _, __ = mod.benchmark_with_tensors(
        inputs,
        ys,
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    print(f"batch_size: {batch_size}, latency: {t}")
    dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    dev_flag = dev_flag.replace(",", "_")
    with open(f"{model_name}_ait_benchmark_dev_{dev_flag}.txt", "a") as f:
        f.write(f"batch_size: {batch_size}, latency: {t}\n")


@click.command()
@click.option("--model-name", type=str, default="vit_base_patch16_224")
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=True,
    help="Whether to use FP16 for accumulation (similar to TensorRT)",
)
@click.option("--use-graph", type=bool, default=True, help="Whether to use CUDA graph")
@click.option("--batch-size", type=int, default=0, help="Batch size")
def main(
    model_name="vit_base_patch16_224", use_fp16_acc=True, use_graph=True, batch_size=0
):
    if detect_target().name() == "rocm":
        use_graph = False
    if batch_size < 1:
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            compile_vit(model_name, bs, use_fp16_acc=use_fp16_acc)
            benchmark(model_name, bs, graph_mode=use_graph)
    else:
        benchmark(model_name, batch_size, graph_mode=use_graph)


if __name__ == "__main__":
    main()
