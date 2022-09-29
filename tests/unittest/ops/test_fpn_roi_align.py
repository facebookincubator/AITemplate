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
import unittest
from unittest import skipIf

import torch
from aitemplate.compiler import compile_model, Model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

try:
    from detectron2.modeling.poolers import ROIPooler

    HAS_D2 = True
except ImportError:
    HAS_D2 = False
skipIfNoD2 = skipIf(not HAS_D2, "no detectron2")


def random_boxes(num_boxes, max_coord=512):
    boxes = torch.rand(num_boxes, 4) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)
    boxes[:, 2:] += boxes[:, :2]
    return boxes.cuda().half()


@skipIfNoD2
class RoiAlignTestCase(unittest.TestCase):
    def _test_fpn_roi_align(
        self,
        boxes,
        features,
        CC=16,
        num_rois=3,
        pooled_size=7,
        spatial_scale=1 / 16.0,
        sampling_ratio=0,
        batch_size=1,
        test_name="fpn_roi_align",
        im_shape=(512, 512),
        rebuild=True,
        bench=False,
    ):
        HH, WW = im_shape
        target = detect_target()

        P2 = Tensor(
            shape=[1, HH // 4, WW // 4, CC], dtype="float16", name="P2", is_input=True
        )

        P3 = Tensor(
            shape=[1, HH // 8, WW // 8, CC], dtype="float16", name="P3", is_input=True
        )
        P4 = Tensor(
            shape=[1, HH // 16, WW // 16, CC], dtype="float16", name="P4", is_input=True
        )
        P5 = Tensor(
            shape=[1, HH // 32, WW // 32, CC], dtype="float16", name="P5", is_input=True
        )
        R = Tensor(shape=[num_rois, 5], dtype="float16", name="ROI", is_input=True)

        OP = ops.multi_level_roi_align(
            num_rois=num_rois,
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            position_sensitive=False,
            continuous_coordinate=True,
            im_shape=im_shape,
        )
        Y = OP(P2, P3, P4, P5, R)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        if rebuild:
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        def fpn_roialign_pt(boxes, features, device="cuda"):
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            from detectron2.structures import Boxes

            pooler_resolution = pooled_size
            canonical_level = 4
            canonical_scale_factor = 2**canonical_level
            pooler_scales = (
                4.0 / canonical_scale_factor,
                2.0 / canonical_scale_factor,
                1.0 / canonical_scale_factor,
                0.5 / canonical_scale_factor,
            )
            sampling_ratio = 0

            rois = [Boxes(boxes).to(device)]

            roialignv2_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type="ROIAlignV2",
            )
            roialignv2_out = roialignv2_pooler(features, rois)
            if bench:
                func = roialignv2_pooler
                args = (features, rois)
                duration = benchmark_torch_function(100, func, *args)
                print(
                    f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
                )

            return roialignv2_out

        y_pt = fpn_roialign_pt(boxes, features)  # fp32 pt

        rois = torch.zeros(num_rois, 5)
        rois[:, 1:] = boxes
        rois = rois.cuda().half()
        X_p2 = features[0].half()
        X_p3 = features[1].half()
        X_p4 = features[2].half()
        X_p5 = features[3].half()

        x_p2 = X_p2.permute((0, 2, 3, 1)).contiguous()
        x_p3 = X_p3.permute((0, 2, 3, 1)).contiguous()
        x_p4 = X_p4.permute((0, 2, 3, 1)).contiguous()
        x_p5 = X_p5.permute((0, 2, 3, 1)).contiguous()

        inputs = {
            "P2": x_p2,
            "P3": x_p3,
            "P4": x_p4,
            "P5": x_p5,
            "ROI": rois,
        }
        y = torch.empty([num_rois, pooled_size, pooled_size, CC]).cuda().half()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        eps = 1e-2
        self.assertTrue(torch.allclose(y_pt.half(), y_transpose, atol=eps, rtol=eps))

    def test_fpn_roi_align(self):
        N, C, H, W = 1, 16, 512, 512
        std = 11
        mean = 0

        feature2 = (torch.rand(N, C, H // 4, W // 4) - 0.5) * 2 * std + mean
        feature3 = (torch.rand(N, C, H // 8, W // 8) - 0.5) * 2 * std + mean
        feature4 = (torch.rand(N, C, H // 16, W // 16) - 0.5) * 2 * std + mean
        feature5 = (torch.rand(N, C, H // 32, W // 32) - 0.5) * 2 * std + mean

        features = [
            feature2.cuda(),
            feature3.cuda(),
            feature4.cuda(),
            feature5.cuda(),
        ]

        boxes = torch.tensor(
            [
                [100.0, 120.0, 152.0, 152.0],
                [2.0, 2.0, 52.0, 52.0],
                [1.0, 1.0, 100.0, 100.0],
                [110.0, 110.0, 300.0, 300.0],
                [1.0, 1.0, 150.0, 150.0],
                [10.0, 10.0, 300.0, 300.0],
                [10.0, 10.0, 400.0, 400.0],
                [110.0, 110.0, 400.0, 400.0],
                [110.0, 110.0, 350.0, 350.0],
                [10.0, 10.0, 510.0, 510.0],
            ]
        ).cuda()
        boxes = random_boxes(100)
        self._test_fpn_roi_align(
            boxes,
            features,
            CC=C,
            num_rois=boxes.shape[0],
            im_shape=(H, W),
            pooled_size=7,
            rebuild=1,
            test_name="fpn_roi_align",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
