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
import timm
import torch
from aitemplate.testing.benchmark_pt import benchmark_torch_function


def benchmark(model, batch_size):
    with torch.inference_mode():
        input_shape = (batch_size, 3, 224, 224)
        input_data = torch.randn(input_shape).cuda().half()
        # warm up
        benchmark_torch_function(100, model, input_data)
        # benchmark
        t = benchmark_torch_function(100, model, input_data)
        print("batch_size: {}, time: {}".format(batch_size, t))
        dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
        dev_flag = dev_flag.replace(",", "_")
        with open(f"resnet50_pt_benchmark_dev_{dev_flag}.txt", "a") as f:
            f.write("batch_size: {}, latency: {}\n".format(batch_size, t))


@click.command()
@click.option("--batch-size", default=0, type=int)
def main(batch_size):
    model = timm.create_model("resnet50", pretrained=False).cuda().half()
    model.eval()
    if batch_size == 0:
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            benchmark(model, batch_size)
    else:
        benchmark(model, batch_size)


if __name__ == "__main__":
    main()
