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
import unittest

import torch

from aitemplate.compiler import compile_model

from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target

from modeling.resnet import build_resnet_backbone
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


class ResNet50Verification(unittest.TestCase):
    def test_resnet50(self):
        target = detect_target()
        batch_size = 1
        torch_dtype = torch.float16
        ait_dtype = "float16"
        # Create input tensor, need to specify the shape, dtype and is_input flag
        x = Tensor(
            shape=[batch_size, 224, 224, 3],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        model = build_resnet_backbone(50, activation="ReLU")
        # Mark all parameters with name same to PyTorch name convention
        model.name_parameter_tensor()
        # Forward the input tensor to the model, get output tensor
        y = model(x)
        # Mark output tensor
        mark_output(y)

        timm_exporter = timm_export("resnet50", pretrained=False)
        ait_params = timm_exporter.export_model(half=torch_dtype == torch.float16)
        pt_model = timm_exporter.pt_model.to(dtype=torch_dtype, device="cuda")
        pt_model.eval()
        module = compile_model(y, target, "./tmp", "resnet50")
        for name, param in ait_params.items():
            module.set_constant_with_tensor(name, param)

        # ait model expects NHWC format
        x_ait = torch.rand([batch_size, 224, 224, 3], dtype=torch_dtype, device="cuda")
        # center the input wrt the training data for numerical stability
        x_ait -= torch.tensor([0.485, 0.456, 0.406]).cuda()
        x_ait /= torch.tensor([0.229, 0.224, 0.225]).cuda()
        # torch model expects NCHW format
        x_pt = torch.transpose(x_ait, 1, 3).contiguous()
        with torch.no_grad():
            y_pt = pt_model(x_pt)
        y_ait = torch.zeros([batch_size, 1, 1, 1000], dtype=torch_dtype, device="cuda")
        module.run_with_tensors([x_ait], [y_ait])

        torch.testing.assert_close(
            y_pt, y_ait.reshape([batch_size, 1000]), rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    unittest.main()
