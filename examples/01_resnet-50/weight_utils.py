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
script for converting model from timm to aitemplate
Only tested on resnet50
"""


import pickle
import re

import click
import numpy as np
import timm
import torch
from aitemplate.testing import detect_target

CONV_WEIGHT_PATTERN = re.compile(r"conv\d+\.weight")


class timm_export:
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        if model_name != "resnet50":
            raise NotImplementedError

        with torch.no_grad():
            self.pt_model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=1000
            )
        self.pt_state = self.pt_model.state_dict()

    def export_model(self, half=True):
        fused_model = {}
        for param_name in self.pt_state.keys():
            self.transform_params(param_name, fused_model)
        ait_model = {k.replace(".", "_"): weight for k, weight in fused_model.items()}
        if detect_target().name() == "cuda":
            self.export_conv0(ait_model, fused_model)
        if half:
            half_params = {}
            for k, v in ait_model.items():
                half_params[k] = v.detach().cuda().half().contiguous()
            return half_params
        return ait_model

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

        # NCHW -> NHWC
        conv_w = conv_w.permute(0, 2, 3, 1).contiguous()
        for arr in [conv_w.numpy(), conv_b.numpy()]:
            if np.isnan(arr).any():
                print("fuse bn error")
        return conv_w, conv_b

    def transform_conv0(self):
        conv_w = self.pt_state["conv1.weight"]
        bn_w = self.pt_state["bn1.weight"]
        bn_b = self.pt_state["bn1.bias"]
        bn_rm = self.pt_state["bn1.running_mean"]
        bn_rv = self.pt_state["bn1.running_var"]
        fused_w, fused_b = self.fuse_conv_bn_weights(
            conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
        )
        return fused_w, fused_b

    def transform_params(self, param_name, fused_model):
        if param_name == "conv1.weight":
            fused_w, fused_b = self.transform_conv0()
            fused_model["stem.conv1.weight"] = fused_w
            fused_model["stem.conv1.bias"] = fused_b
        elif "downsample.0.weight" in param_name:
            fused_w, fused_b = self.transform_downsample(param_name)
            fused_model[param_name] = fused_w
            fused_model[param_name.replace("weight", "bias")] = fused_b
        elif param_name == "fc.weight":
            fused_model["fc.weight"] = self.pt_state["fc.weight"]
            fused_model["fc.bias"] = self.pt_state["fc.bias"]
        elif CONV_WEIGHT_PATTERN.search(param_name) is not None:
            bn_w_name = param_name.replace("conv", "bn")
            conv_w = self.pt_state[param_name]
            bn_w = self.pt_state[bn_w_name]
            bn_b = self.pt_state[bn_w_name.replace("weight", "bias")]
            bn_rm = self.pt_state[bn_w_name.replace("weight", "running_mean")]
            bn_rv = self.pt_state[bn_w_name.replace("weight", "running_var")]
            fused_w, fused_b = self.fuse_conv_bn_weights(
                conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
            )
            fused_model[param_name] = fused_w
            fused_model[param_name.replace("weight", "bias")] = fused_b
        else:
            pass

    def transform_downsample(self, param_name):
        assert "downsample" in param_name
        tags = param_name.split(".")
        block_tag = ".".join(tags[:2])
        conv_w = self.pt_state[f"{block_tag}.downsample.0.weight"]
        bn_w = self.pt_state[f"{block_tag}.downsample.1.weight"]
        bn_b = self.pt_state[f"{block_tag}.downsample.1.bias"]
        bn_rm = self.pt_state[f"{block_tag}.downsample.1.running_mean"]
        bn_rv = self.pt_state[f"{block_tag}.downsample.1.running_var"]
        fused_w, fused_b = self.fuse_conv_bn_weights(
            conv_w, None, bn_rm, bn_rv, 1e-5, bn_w, bn_b
        )
        return fused_w, fused_b

    def export_conv0(self, ait_model, fuse_model):
        pt_name = "stem.conv1.weight"
        x = fuse_model[pt_name]
        conv_w = torch.zeros((64, 7, 7, 4))
        conv_w[:, :, :, :3] = x
        ait_model[pt_name.replace(".", "_")] = conv_w


def export_to_torch_tensor(model_name="resnet50"):
    if model_name != "resnet50":
        raise NotImplementedError
    timm2ait = timm_export(model_name)
    ait_model = timm2ait.export_model(half=True)
    return ait_model


@click.command()
@click.option("--param-path", type=str, default="resnet50.pkl")
def export_to_numpy(param_path):
    ait_model = export_to_torch_tensor()
    np_weights = {}
    for k, v in ait_model.items():
        np_weights[k] = v.detach().cpu().numpy().astype(np.float16)

    with open(param_path, "wb") as f:
        pickle.dump(np_weights, f)


if __name__ == "__main__":
    export_to_numpy()
