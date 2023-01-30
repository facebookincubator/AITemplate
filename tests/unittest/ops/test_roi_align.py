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
from unittest import skipIf

import numpy as np
import torch
from aitemplate.compiler import compile_model

from aitemplate.frontend import IntVar, nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


def random_boxes(num_boxes, max_coord=100):
    boxes = torch.rand(num_boxes, 4) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


@skipIfNoTorchVision
class RoiAlignTestCase(unittest.TestCase):
    def _create_tensors(self, num_rois, b, rand=False, dtype="float16"):
        if rand:
            boxes = random_boxes(num_rois, 200)
            inds = np.arange(b)
            batch_inds = np.repeat(inds.reshape(-1, 1), repeats=num_rois // b, axis=1)
            rois = torch.cat(
                (torch.tensor(batch_inds).reshape(b, -1, 1), boxes.reshape(b, -1, 4)), 2
            )
            rois = rois.reshape(-1, 5).cuda()
        else:
            rois = torch.tensor(
                [
                    [0, -2.0, -2.0, 22.0, 22.0],
                    [0, 10.0, 10.0, 30.0, 30.0],
                    [0, 1.0, 1.0, 10.0, 10.0],
                    [1, -2.0, -2.0, 22.0, 22.0],
                    [1, 10.0, 10.0, 30.0, 30.0],
                    [1, 1.0, 1.0, 10.0, 10.0],
                ]
            ).cuda()
        torch_dtype = string_to_torch_dtype(dtype)
        return rois.to(dtype=torch_dtype)

    def _test_single_op(
        self,
        HH,
        WW,
        CC,
        num_rois=3,
        pooled_size=7,
        spatial_scale=1 / 16.0,
        sampling_ratio=2,
        batch_size=(1, 1),
        rand=False,
        test_name="roi_align_fp16",
        dtype="float16",
    ):
        target = detect_target()

        X = Tensor(
            shape=[IntVar(values=list(batch_size), name="input_batch"), HH, WW, CC],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        R = Tensor(
            shape=[IntVar(values=[num_rois, num_rois], name="roi_batch"), 5],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        output_size = (pooled_size, pooled_size)

        OP = nn.RoiAlign(
            num_rois=num_rois,
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            position_sensitive=False,
            continuous_coordinate=False,
        )
        Y = OP(X, R)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, CC, WW, HH], dtype=dtype)
            rois = self._create_tensors(num_rois, b, rand, dtype=dtype)

            if b == 1:
                rois = rois[:num_rois, :]

            OP_pt = torchvision.ops.RoIAlign(
                output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio
            )
            Y_pt = OP_pt(X_pt, rois)
            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors([x, rois], [y])
            y_transpose = y.permute((0, 3, 1, 2))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-1, rtol=1e-1))

    def test_roi_align_fp16(self):
        self._test_single_op(
            HH=56,
            WW=56,
            CC=256,
            test_name="roi_align1",
            dtype="float16",
        )
        # self._test_fp16_single_op(HH=16, WW=16, CC=32, num_rois=6, batch_size=(2, 2), rand=True, test_name="roi_align2")

    def test_roi_align_fp32(self):
        self._test_single_op(
            HH=56,
            WW=56,
            CC=256,
            test_name="roi_align1",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
