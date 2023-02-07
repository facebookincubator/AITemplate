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
"""benchmark for resnet50"""

import os

import click

import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from modeling.resnet import build_resnet_backbone
from weight_utils import export_to_torch_tensor


def mark_output(y):
    """Different to PyTorch, we need to explicit mark output tensor for optimization,

    Parameters
    ----------
    y : List[Tensor]
        List of output tensors
    """
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def compile_module(model_name, batch_size, **kwargs):

    if model_name != "resnet50":
        raise NotImplementedError

    model_name = f"{model_name}_{batch_size}"
    target = detect_target(**kwargs)
    # Create input tensor, need to specify the shape, dtype and is_input flag
    x = Tensor(
        shape=[batch_size, 224, 224, 3], dtype="float16", name="input0", is_input=True
    )
    model = build_resnet_backbone(50, activation="ReLU")
    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(x)
    # Mark output tensor
    mark_output(y)
    # Compile the model
    module = compile_model(y, target, "./tmp", model_name)
    return module


def benchmark(model_name, batch_size, mod=None, graph_mode=True):
    # Load params
    cuda_params = export_to_torch_tensor(model_name)
    # Load compiled model
    if mod is None:
        model_name = f"{model_name}_{batch_size}"
        mod = Model(os.path.join("./tmp", model_name, "test.so"))

    # Set params
    mod.set_many_constants_with_tensors(cuda_params)
    mod.fold_constants(sync=True)

    # prepare input/output tensor
    x_input = torch.randn([batch_size, 224, 224, 3]).cuda().half()
    x_input = x_input.contiguous()
    y_output = torch.zeros([batch_size, 1, 1, 1000]).cuda().half()
    y_output = y_output.contiguous()

    # warm up
    t, _, __ = mod.benchmark_with_tensors(
        [x_input],
        [y_output],
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    # benchmark
    t, _, __ = mod.benchmark_with_tensors(
        [x_input],
        [y_output],
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    print(f"batch_size: {batch_size}, latency: {t}")
    dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    dev_flag = dev_flag.replace(",", "_")
    with open(f"resnet50_ait_benchmark_dev_{dev_flag}.txt", "a") as f:
        f.write(f"batch_size: {batch_size}, latency: {t}\n")


@click.command()
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=True,
    help="Whether to use FP16 for accumulation (similar to TensorRT)",
)
@click.option("--use-graph", type=bool, default=True, help="Whether to use CUDA graph")
@click.option("--batch-size", type=int, default=0, help="Batch size")
def main(use_fp16_acc=True, use_graph=True, batch_size=0):
    if detect_target().name() == "rocm":
        use_graph = False
    if batch_size < 1:
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            compile_module("resnet50", bs, use_fp16_acc=use_fp16_acc)
            benchmark("resnet50", bs, graph_mode=use_graph)
    else:
        benchmark("resnet50", batch_size, graph_mode=use_graph)


if __name__ == "__main__":
    main()
