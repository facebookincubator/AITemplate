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
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from modeling.torch_model import BertBaseUncased


def benchmark_pt(pretrained=True, batchsize=0):
    bert = BertBaseUncased(pretrained=pretrained)
    model = bert._model
    model.eval()

    if batchsize == 0:
        candidate_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        candidate_batch_sizes = [batchsize]

    with torch.inference_mode():
        for seq_length in [64, 128, 384, 512]:
            for batch_size in candidate_batch_sizes:
                try:
                    input_ids, token_type_ids, position_ids = bert.generate_inputs(
                        batch_size, seq_length
                    )
                    bert.forward(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                    )
                    # warmup
                    t = benchmark_torch_function(
                        100,
                        bert.forward,
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                    )
                    # benchmark
                    t = benchmark_torch_function(
                        100,
                        bert.forward,
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                    )
                    print(
                        f"bert pt: batch_size: {batch_size}, seq_length: {seq_length}, {t} ms",
                    )
                    with open("bert_pt_benchmark.txt", "a") as f:
                        f.write(
                            f"batch_size: {batch_size}, seq_length: {seq_length} latency: {t} ms\n"
                        )
                except RuntimeError:
                    # pt runs out of memory
                    break


def benchmark_pt_encoders_only(pretrained=True, batchsize=0):
    model = BertBaseUncased(pretrained=pretrained)
    pt_bert = model._model
    pt_bert.eval()

    encoder = pt_bert.bert.encoder
    hidden_size = pt_bert.config.hidden_size

    if batchsize == 0:
        candidate_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        candidate_batch_sizes = [batchsize]

    for seq_length in [64, 128, 384, 512, 1024, 4096]:
        for batch_size in candidate_batch_sizes:
            try:
                encoder_input = (
                    torch.randn([batch_size, seq_length, hidden_size]).cuda().half()
                )
                encoder.forward(encoder_input)
                # warmup
                t = benchmark_torch_function(
                    100,
                    encoder.forward,
                    encoder_input,
                )
                # benchmark
                t = benchmark_torch_function(
                    100,
                    encoder.forward,
                    encoder_input,
                )
                print(
                    f"bert encoders pt: batch_size: {batch_size}, seq_length: {seq_length}, {t} ms",
                )
                with open("bert_encoders_pt_benchmark.txt", "a") as f:
                    f.write(
                        f"batch_size: {batch_size}, seq_length: {seq_length} latency: {t} ms\n"
                    )
            except RuntimeError:
                # pt runs out of memory
                break


@click.command()
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
@click.option(
    "--batch-size",
    type=int,
    default=0,
    help="The batch size to use for the benchmark. If 0, the batch size is default [1 : 128].",
)
def benchmark(
    use_pretrained_pt_model: bool,
    encoders_only: bool,
    batch_size: int,
):
    if encoders_only:
        benchmark_pt_encoders_only(use_pretrained_pt_model, batch_size)
    else:
        benchmark_pt(use_pretrained_pt_model, batch_size)


if __name__ == "__main__":
    torch.manual_seed(4896)
    benchmark()
