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

from typing import Dict, List

import click
import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from modeling.bert import BertBaseEncodersOnly, BertBaseUncased
from modeling.torch_model import BertBaseUncased as BertPt


def mark_output(y: Tensor) -> None:
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def create_bert_inputs(
    batch_size: int, seq_length: int, dtype: str = "int64"
) -> List[Tensor]:
    input_ids = Tensor(
        shape=[batch_size, seq_length],
        name="input_ids",
        dtype=dtype,
        is_input=True,
    )
    token_type_ids = Tensor(
        shape=[batch_size, seq_length],
        name="token_type_ids",
        dtype=dtype,
        is_input=True,
    )
    position_ids = Tensor(
        shape=[batch_size, seq_length],
        name="position_ids",
        dtype=dtype,
        is_input=True,
    )
    return [input_ids, token_type_ids, position_ids]


def create_bert_encoders_input(
    batch_size: int, seq_length: int, hidden: int, dtype: str = "float16"
):
    encoder_input = Tensor(
        shape=[batch_size, seq_length, hidden],
        name="input",
        dtype=dtype,
        is_input=True,
    )
    return [encoder_input]


def create_bert_inputs_pt(
    batch_size: int,
    seq_length: int,
    vocab_size: int = 30522,
    type_vocab_size: int = 2,
    dtype: torch.dtype = torch.int64,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_length), dtype=dtype
    ).cuda()
    token_type_ids = torch.randint(
        0, type_vocab_size, input_ids.size(), dtype=dtype
    ).cuda()
    position_ids = (
        torch.arange(seq_length, dtype=dtype)
        .reshape((1, -1))
        .expand(batch_size, -1)
        .contiguous()
        .cuda()
    )
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def create_bert_encoders_inputs_pt(
    batch_size: int, seq_length: int, hidden_size: int
) -> Dict[str, torch.Tensor]:
    encoder_input = torch.randn([batch_size, seq_length, hidden_size]).cuda().half()
    return {"input": encoder_input}


def map_pt_params(
    ait_bert, pt_bert, batch_size: int, seq_length: int
) -> Dict[str, torch.Tensor]:
    pt_params = dict(pt_bert.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_bert.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            mapped_pt_params[ait_name] = pt_params[name]
            continue

        if name.endswith("self.qkv.weight"):
            prefix = name[: -len("qkv.weight")]
            q_weight = pt_params[prefix + "query.weight"]
            k_weight = pt_params[prefix + "key.weight"]
            v_weight = pt_params[prefix + "value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_pt_params[ait_name] = qkv_weight
        elif name.endswith("self.qkv.bias"):
            prefix = name[: -len("qkv.bias")]
            q_bias = pt_params[prefix + "query.bias"]
            k_bias = pt_params[prefix + "key.bias"]
            v_bias = pt_params[prefix + "value.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            mapped_pt_params[ait_name] = qkv_bias
        elif name.endswith("self.proj.weight"):
            prefix = name[: -len("self.proj.weight")]
            pt_name = prefix + "output.dense.weight"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("self.proj.bias"):
            prefix = name[: -len("self.proj.bias")]
            pt_name = prefix + "output.dense.bias"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("cu_length"):
            cu_len = np.cumsum([0] + [seq_length] * batch_size).astype("int32")
            mapped_pt_params[ait_name] = torch.from_numpy(cu_len).cuda()
        else:
            pt_param = pt_bert.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param

    return mapped_pt_params


def benchmark(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    mod: Model,
    graph_mode: bool,
    encoders_only: bool,
):
    if encoders_only:
        inputs = create_bert_encoders_inputs_pt(batch_size, seq_length, hidden_size)
    else:
        inputs = create_bert_inputs_pt(batch_size, seq_length)

    outputs = [torch.empty(mod.get_output_maximum_shape(0)).cuda().half()]

    # warm up
    t, _, __ = mod.benchmark_with_tensors(
        inputs,
        outputs,
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    # benchmark
    t, _, __ = mod.benchmark_with_tensors(
        inputs,
        outputs,
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    print(f"batch_size: {batch_size}, seq_length: {seq_length}, latency: {t}")
    dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    dev_flag = dev_flag.replace(",", "_")
    with open(f"bert_ait_benchmark_dev_{dev_flag}.txt", "a") as f:
        f.write(f"batch_size: {batch_size}, seq_length: {seq_length}, latency: {t}\n")


def compile_module(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    activation: str,
    use_fp16_acc: bool,
    encoders_only: bool,
    pt_model: torch.nn.Module,
) -> None:
    model_name = f"BERT_{activation}_{batch_size}_{seq_length}"
    target = detect_target(use_fp16_acc=use_fp16_acc)

    if encoders_only:
        inputs = create_bert_encoders_input(batch_size, seq_length, hidden_size)
    else:
        inputs = create_bert_inputs(batch_size, seq_length)

    if encoders_only:
        model = BertBaseEncodersOnly(batch_size, seq_length, hidden_act=activation)
    else:
        model = BertBaseUncased(batch_size, seq_length, hidden_act=activation)

    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(*inputs)
    # Mark output tensor
    mark_output(y)

    params = map_pt_params(model, pt_model, batch_size, seq_length)

    mod = compile_model(y, target, "./tmp", model_name)

    mod.set_many_constants_with_tensors(params)
    mod.fold_constants(sync=True)

    return mod




@click.command()
@click.option("--batch-size", type=int, default=0, help="Inference batch size")
@click.option("--seq-length", type=int, default=0, help="Inference sequence length")
@click.option(
    "--activation",
    type=str,
    default="fast_gelu",
    help="Activation function applied on BERT, currently only support fast_gelu on Rocm. CUDA supports both gelu and fast_gelu. No effect if framework is pt.",
)
@click.option(
    "--graph-mode",
    type=bool,
    default=True,
    help="Use CUDA graph or not. hipGraph is not supported yet. No effect if framework is pt.",
)
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=True,
    help="Use fp16 accumulation or not (TensorRT is using fp16_acc). No effect if framework is pt.",
)
@click.option(
    "--use-pretrained-pt-model",
    type=bool,
    default=True,
    help="Whether or not to use the pretrained BERT model weights.",
)
@click.option(
    "--encoders-only",
    type=bool,
    default=True,
    help="Whether or not to run the BERT benchmark with encoders only. If enabled, only the transformer blocks without BERT embeddings are benchmarked.",
)
def compile_and_benchmark(
    batch_size: int,
    seq_length: int,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    use_pretrained_pt_model: bool,
    encoders_only: bool,
):
    if detect_target().name() == "rocm":
        graph_mode = False
        assert activation in (
            "fast_gelu"
        ), f"Unsupported activation: {activation} on rocm"

    pt_model = BertPt(pretrained=use_pretrained_pt_model)._model
    pt_model.eval()
    hidden_size = pt_model.config.hidden_size

    if batch_size < 1:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        batch_sizes = [batch_size]

    if seq_length < 1:
        seq_lengths = (
            [64, 128, 384, 512, 1024, 4096] if encoders_only else [64, 128, 384, 512]
        )
    else:
        seq_lengths = [seq_length]

    for seq_length in seq_lengths:
        for bs in batch_sizes:
            mod = compile_module(
                bs,
                seq_length,
                hidden_size,
                activation,
                use_fp16_acc,
                encoders_only,
                pt_model,
            )
            benchmark(bs, seq_length, hidden_size, mod, graph_mode, encoders_only)


if __name__ == "__main__":
    torch.manual_seed(4896)
    compile_and_benchmark()
