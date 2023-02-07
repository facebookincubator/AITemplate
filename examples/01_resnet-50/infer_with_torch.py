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

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from modeling.resnet import build_resnet_backbone
from PIL import Image
from weight_utils import timm_export


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


def compile_module(model_name, **kwargs):
    batch_size = 1

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


def prepare_data(img_path=None):
    # we find a 224x224 image online for demo purpose:
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    if img_path is None:
        if os.path.exists("cat.png") is False:
            os.system(f"wget -O cat.png {img_url}")
        img_path = "cat.png"
    image = Image.open(img_path).resize((224, 224))
    image = torch.as_tensor(np.array(image).astype("float32")).cuda().half()
    image = image.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().half()
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().half()
    image = (image / 255.0 - mean[None, None, None, :]) / std[None, None, None, :]
    return image


def export_to_torch_tensor(model_name="resnet50"):
    if model_name != "resnet50":
        raise NotImplementedError
    timm2ait = timm_export(model_name)
    params = timm2ait.export_model(half=True)
    return params, timm2ait.pt_model


def inference(model_name, mod=None):
    # Load params
    cuda_params, pt_model = export_to_torch_tensor(model_name)
    # Load compiled model
    if mod is None:
        mod = Model(os.path.join("./tmp", model_name, "test.so"))

    # Set torch tensor params to runtime
    mod.set_many_constants_with_tensors(cuda_params)
    mod.fold_constants(sync=True)

    # prepare input/output tensor
    x_input = prepare_data()
    x_input = x_input.contiguous()
    y_output = torch.zeros([1, 1, 1, 1000]).cuda().half()
    y_output = y_output.contiguous()

    # execute
    mod.run_with_tensors([x_input], [y_output])

    # process output with pytorch
    y_label = torch.argmax(y_output, dim=-1)
    y_cpu = y_label.cpu().numpy()
    print(y_cpu)

    # run pytorch
    pt_model.eval()
    pt_model = pt_model.cuda().half()
    pt_output = pt_model(x_input.permute([0, 3, 1, 2]))
    y_label = torch.argmax(pt_output, dim=-1)
    y_cpu = y_label.cpu().numpy()
    print(y_cpu)

    # verify outputs
    assert torch.allclose(y_output, pt_output, 1e-1, 1e-1)
    print("Verification done!")


if __name__ == "__main__":
    np.random.seed(4896)
    model_name = "resnet50"
    mod = compile_module(model_name, use_fp16_acc=True)
    inference(model_name, mod)
