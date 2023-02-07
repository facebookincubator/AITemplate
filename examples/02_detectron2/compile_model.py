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
import json
import os

import click

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from configs import get_cfg_defaults
from modeling.meta_arch import GeneralizedRCNN

# pylint: disable=W0102


def rand_init(shape):
    if len(shape) == 1:
        arr = np.zeros(shape).astype("float16")
    else:
        fout = shape[0]
        fin = shape[-1]
        scale = np.sqrt(2) / np.sqrt(fout + fin)
        arr = np.random.normal(0, scale, shape).astype("float16")
    return torch.from_numpy(arr).cuda().half()


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def extract_params_meta(net):
    ret = []
    params = net.parameters()
    for p in params:
        t = p.tensor()
        name = t._attrs["name"]
        shape = [x._attrs["values"][0] for x in t._attrs["shape"]]
        ret.append([name, shape])
    return ret


def benchmark(cfg, mod=None):
    im_shape = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, 3)
    HH, WW, CC = im_shape
    BS = cfg.SOLVER.IMS_PER_BATCH
    inputs = np.random.normal(0, 1, (BS, HH, WW, CC)).astype("float16")

    model_name = cfg.MODEL.NAME
    if mod is None:
        mod = Model(os.path.join("./tmp", model_name, "test.so"))

    ait_mod = GeneralizedRCNN(cfg)

    for name, param in ait_mod.named_parameters():
        shape = get_shape(param.tensor())
        arr = rand_init(shape)
        mod.set_constant_with_tensor(name.replace(".", "_"), arr)

    x_input = torch.tensor(inputs).cuda().half()
    x = x_input.contiguous()

    GeneralizedRCNN(cfg).set_anchors(mod)

    topk = cfg.POSTPROCESS.TOPK
    outputs = [
        torch.empty([BS, 1], dtype=torch.int64).cuda(),
        torch.empty([BS, topk, 4]).cuda().half(),
        torch.empty([BS, topk]).cuda().half(),
        torch.empty([BS, topk], dtype=torch.int64).cuda(),
    ]
    if cfg.MODEL.MASK_ON:
        mask_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
        outputs.append(torch.empty([BS, topk, mask_size, mask_size]).cuda().half())

    mod.fold_constants(sync=True)
    mod.benchmark_with_tensors([x], outputs, count=100, repeat=2, graph_mode=True)


def compile_module(cfg):
    model_name = cfg.MODEL.NAME
    target = detect_target()

    im_shape = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, 3)
    HH, WW, CC = im_shape
    BS = cfg.SOLVER.IMS_PER_BATCH
    x = Tensor(shape=[BS, HH, WW, CC], dtype="float16", name="input_0", is_input=True)
    model = GeneralizedRCNN(cfg)
    model.name_parameter_tensor()

    y = model(x)
    mark_output(y)
    module = compile_model(y, target, "./tmp", model_name)

    with open(os.path.join("./tmp", model_name, "params.json"), "w") as fo:
        fo.write(json.dumps(extract_params_meta(model)))

    benchmark(cfg, module)


@click.command()
@click.option("--config", default="", metavar="FILE", help="path to config file")
@click.option("--bench-config", default="", metavar="FILE", help="path to config file")
@click.option("--batch", default=0, help="batch size")
@click.option("--eval/--no-eval", default=False, help="perform evaluation only")
def compile_and_benchmark(config, bench_config, batch, eval):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    if bench_config != "":
        cfg.merge_from_file(bench_config)
    if batch > 0:
        cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.freeze()
    print(cfg.MODEL.NAME)

    if eval:
        benchmark(cfg)
    else:
        compile_module(cfg)


if __name__ == "__main__":
    np.random.seed(4896)
    compile_and_benchmark()
