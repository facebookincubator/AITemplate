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

from transformers import BertTokenizer

from .benchmark_ait import compile_module
from .modeling.torch_model import BertBaseUncased as BertPt


def prepare_data(prompt: str, model_path: str):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    result = tokenizer(prompt, return_attention_mask=False, return_tensors="pt")
    target_size = result["input_ids"].size()
    if target_size[1] > 512:
        raise ValueError("Sequence length > 512 is not supported")

    result["position_ids"] = (
        torch.arange(target_size[1], dtype=torch.int64)
        .reshape(result["input_ids"].size())
        .contiguous()
        .cuda()
    )
    return result


def run_model(
    prompt: str,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    verify: bool,
    model_path="bert-base-uncased",
):
    inputs = prepare_data(prompt, model_path)
    inputs_pt = {name: data.cuda() for name, data in inputs.items()}
    batch_size, seq_len = inputs["input_ids"].size()

    pt_model = BertPt(model_path=model_path, pretrained=True)._model
    pt_model.eval()
    hidden_size = pt_model.config.hidden_size

    mod = compile_module(
        batch_size, seq_len, hidden_size, activation, use_fp16_acc, False, pt_model
    )

    outputs = [torch.empty(mod.get_output_maximum_shape(0)).half().cuda()]
    mod.run_with_tensors(inputs_pt, outputs, graph_mode=graph_mode)

    print(f"Logits: {outputs[0]}")
    if verify:
        pt_outputs = pt_model.bert(**inputs_pt)
        torch.allclose(outputs[0], pt_outputs.last_hidden_state, 1e-1, 1e-1)
        print("Verification done!")


@click.command()
@click.option(
    "--prompt",
    type=str,
    default="The quick brown fox jumps over the lazy dog.",
    help="The prompt to give BERT.",
)
@click.option(
    "--activation",
    type=str,
    default="fast_gelu",
    help="Activation function applied on BERT, currently only support gelu and fast_gelu",
)
@click.option(
    "--graph_mode",
    type=bool,
    default=True,
    help="Use CUDA graph or not. (hipGraph is not supported yet)",
)
@click.option(
    "--use_fp16_acc",
    type=bool,
    default=True,
    help="Use fp16 accumulation or not (TensorRT is using fp16_acc)",
)
@click.option(
    "--verify",
    type=bool,
    default=True,
    help="Verify AIT outputs against PT",
)
def run_demo(
    prompt: str,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    verify: bool,
):
    run_model(prompt, activation, graph_mode, use_fp16_acc, verify)


if __name__ == "__main__":
    torch.manual_seed(4896)
    run_demo()
