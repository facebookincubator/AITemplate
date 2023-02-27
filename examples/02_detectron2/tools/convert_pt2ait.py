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
script for converting model from detectron2 to aitemplate
"""

import json
import os
import pickle as pkl

import click

import numpy as np
import torch
from aitemplate.testing import detect_target

# pylint: disable=C0103


class detectron2_export:
    def __init__(self, model_name):
        self.model_name = model_name

    def export_model(self, model, ait_param_map=None):
        fuse_model = {}
        bn_keys = set()
        for k, _ in model.items():
            if "norm" in k:
                param_name = k.split(".norm")[0]
                if param_name in bn_keys:
                    continue
                bn_keys.add(param_name)
                self.transform_params(param_name, model, fuse_model, fuse_bn=True)
            else:
                self.transform_params(k, model, fuse_model, fuse_bn=False)

        ait_model = {
            k.replace(".", "_"): weight
            for k, weight in fuse_model.items()
            if "anchors" not in k
        }

        if detect_target().name() == "cuda":
            self.export_conv0(ait_model, fuse_model)

        self.check_model(ait_model, ait_param_map)
        return ait_model

    def check_model(self, ait_model, param_map=None):
        if param_map is None:
            with open(os.path.join("./tmp", self.model_name, "params.json")) as fi:
                param_map = json.load(fi)
        for name, shape in param_map:
            assert ait_model[name].shape == tuple(
                shape
            ), "weight shape mismatch {} {} expected {}".format(
                name, ait_model[name].shape, shape
            )

    def fuse_conv_bn_weights(
        self, conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False
    ):
        conv_w = torch.tensor(conv_w)
        bn_rm = torch.tensor(bn_rm)
        bn_rv = torch.tensor(bn_rv)
        bn_w = torch.tensor(bn_w)
        bn_b = torch.tensor(bn_b)
        bn_eps = torch.tensor(bn_eps)

        if conv_b is None:
            conv_b = torch.zeros_like(bn_rm)
        if bn_w is None:
            bn_w = torch.ones_like(bn_rm)
        if bn_b is None:
            bn_b = torch.zeros_like(bn_rm)
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        if transpose:
            shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
        else:
            shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
        conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        for arr in [conv_w.numpy(), conv_b.numpy()]:
            if np.isnan(arr).any():
                print("fuse bn error")
        return conv_w, conv_b

    def transform_params(self, param_name, obj, fuse_model, fuse_bn=True):
        if not fuse_bn:
            arr = obj[param_name]
            if len(arr.shape) == 4:
                arr = np.transpose(arr, (0, 2, 3, 1))
            elif "fc1.weight" in param_name:
                arr = arr.reshape((1024, -1, 7, 7))
                arr = np.transpose(arr, (0, 2, 3, 1))
                arr = arr.reshape((1024, -1))
            fuse_model[param_name] = torch.tensor(arr)

        else:
            conv_k = "%s.weight" % (param_name)
            conv_b = "%s.bias" % (param_name)
            bn_w_k = "%s.norm.weight" % (param_name)
            bn_b_k = "%s.norm.bias" % (param_name)
            bn_rm_k = "%s.norm.running_mean" % (param_name)
            bn_rv_k = "%s.norm.running_var" % (param_name)
            fused_conv_weight, fused_conv_bias = self.fuse_conv_bn_weights(
                obj[conv_k],
                None,
                obj[bn_rm_k],
                obj[bn_rv_k],
                1e-5,
                obj[bn_w_k],
                obj[bn_b_k],
            )
            fuse_model[conv_k] = fused_conv_weight.permute((0, 2, 3, 1))
            fuse_model[conv_b] = fused_conv_bias

    def export_conv0(self, ait_model, fuse_model):
        pt_name = "backbone.bottom_up.stem.conv1.weight"
        x = fuse_model[pt_name]
        conv_w = torch.zeros((64, 7, 7, 4))
        conv_w[:, :, :, :3] = x
        ait_model[pt_name.replace(".", "_")] = conv_w


@click.command()
@click.option("--model-name", default="", metavar="FILE", help="path to ait param file")
@click.option("--d2-weight", default="", metavar="FILE", help="D2 weight")
@click.option("--ait-weight", default="", metavar="FILE", help="AIT weight")
def export_pt_model_to_ait(model_name, d2_weight, ait_weight):
    d2ait = detectron2_export(model_name)
    with open(d2_weight, "rb") as f:
        file = f.read()
    obj = pkl.loads(file, encoding="latin1")
    pt_model = obj["model"]

    ait_model = d2ait.export_model(pt_model)

    torch.save(ait_model, ait_weight)


if __name__ == "__main__":
    export_pt_model_to_ait()
