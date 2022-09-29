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
Unittests for nms Operator.
"""
import unittest
from unittest import skipIf

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

try:
    from torchvision.ops import boxes as box_ops

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


def nonempty(box, threshold=0.0):
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths < threshold) | (heights < threshold)
    return keep


def create_tensors(N):
    dets = np.array(
        [
            [1.5862e02, 1.6100e02, 4.2800e02, 3.9400e02, 7.7100e-01],
            [1.5162e02, 1.5938e02, 4.2800e02, 4.0100e02, 9.2676e-01],
            [1.4700e02, 1.6175e02, 4.3050e02, 3.9925e02, 7.8516e-01],
            [1.4688e02, 1.6038e02, 4.3150e02, 4.0050e02, 8.5498e-01],
            [1.4912e02, 1.6000e02, 4.3150e02, 3.9750e02, 7.0020e-01],
            [1.4925e02, 2.7775e02, 2.4175e02, 3.4775e02, 5.4053e-01],
            [0.0000e00, 2.0250e02, 5.1200e02, 4.9900e02, 2.5223e-02],
            [1.5250e02, 1.5900e02, 4.3100e02, 3.9300e02, 6.5674e-01],
            [1.5262e02, 1.6125e02, 4.3300e02, 3.9475e02, 6.2646e-01],
            [1.5362e02, 1.5375e02, 4.5000e02, 3.9125e02, 7.8857e-02],
            [0.0000e00, 8.8875e01, 5.1200e02, 4.9050e02, 4.6120e-03],
            [1.5000e02, 1.5700e02, 4.2800e02, 3.9900e02, 8.8672e-01],
            [1.5712e02, 1.6150e02, 4.2850e02, 3.9850e02, 9.1162e-01],
            [1.5412e02, 1.6050e02, 4.2650e02, 3.9700e02, 7.0654e-01],
            [1.5112e02, 2.8100e02, 2.4688e02, 3.4700e02, 4.1577e-01],
            [1.3862e02, 1.7175e02, 4.2450e02, 4.0675e02, 4.4495e-02],
            [1.5275e02, 1.6350e02, 4.3175e02, 3.9700e02, 8.6182e-01],
            [1.4875e02, 1.5950e02, 4.2875e02, 3.9700e02, 8.0908e-01],
            [1.4850e02, 1.6000e02, 4.3900e02, 4.0100e02, 6.3965e-01],
            [1.4375e02, 1.2675e02, 4.6275e02, 3.8525e02, 6.5689e-03],
            [0.0000e00, 2.7700e02, 4.6600e02, 4.6800e02, 2.2247e-02],
            [1.6250e00, 4.7650e02, 7.2812e01, 5.0900e02, 5.0430e-03],
            [1.4975e02, 1.6500e02, 4.3125e02, 3.9850e02, 8.2031e-01],
            [1.4950e02, 2.7625e02, 2.7125e02, 3.6025e02, 1.2842e-01],
            [1.5475e02, 1.5788e02, 4.3575e02, 3.9900e02, 8.3789e-01],
            [2.5925e02, 1.7750e01, 5.0925e02, 3.2475e02, 1.0967e-03],
            [2.6200e02, 3.2812e00, 4.9500e02, 7.5375e01, 2.4612e-02],
            [3.3000e01, 1.1462e02, 5.1200e02, 4.6850e02, 3.6469e-03],
            [1.4962e02, 1.6250e02, 4.3650e02, 3.9800e02, 7.9492e-01],
            [1.4850e02, 1.5975e02, 4.3250e02, 3.9275e02, 2.7051e-01],
        ],
        dtype="float16",
    )

    return dets[:N, :4], dets[:N, -1]


@skipIfNoTorchVision
class nmsTestCase(unittest.TestCase):
    def _create_tensors(self, N):
        boxes, scores = create_tensors(N)

        return torch.tensor(boxes).cuda().half(), torch.tensor(scores).cuda().half()

    def test_nms(
        self,
        N=30,
        preNmsTop=30,
        nmsMaxOut=10,
        iouThreshold=0.5,
        minBoxSize=0,
        num_classes=1,
        test_name="proposal_nms",
    ):
        target = detect_target()

        X1 = Tensor(
            shape=[1, N, 4],
            dtype="float16",
            name="X",
            is_input=True,
        )

        X2 = Tensor(
            shape=[1, N],
            dtype="float16",
            name="kernel",
            is_input=True,
        )

        X4 = ops.nms(
            preNmsTop=preNmsTop,
            nmsMaxOut=nmsMaxOut,
            iouThreshold=iouThreshold,
            minBoxSize=minBoxSize,
        )(X1, X2)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        module = compile_model(X4, target, "./tmp", test_name)

        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,)).cuda().half()
        iou = iouThreshold
        kept = nonempty(boxes, threshold=minBoxSize)
        score_pt = scores.clone()
        score_pt[kept] = -1
        keep = box_ops.batched_nms(boxes, score_pt, idxs, iou)

        if keep.shape[0] >= nmsMaxOut:
            keep = keep[:nmsMaxOut]
            ref_box = boxes[keep]
        else:
            ref_box = torch.zeros(nmsMaxOut, 4).half()
            ref_box[
                : keep.shape[0],
            ] = boxes[keep]

        x = boxes.reshape((1, N, 4)).contiguous()
        x_scores = scores.reshape((1, N)).contiguous()
        inputs = [x, x_scores]
        y = torch.empty([1, nmsMaxOut, 4]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(ref_box.cuda(), y, atol=1e-2, rtol=1e-2))

    def test_topk_nms(self, batch_size=1, N=30, topK=30, iou=0.5, test_name="topk_nms"):

        target = detect_target()
        if target.name() == "rocm":
            return
        m_shape = (N, 4)

        def model():
            X_boxes = Tensor(
                shape=m_shape,
                dtype="float16",
                name="X",
                is_input=True,
            )
            X_scores = Tensor(
                shape=[N],
                dtype="float16",
                name="scores",
                is_input=True,
            )
            score_inds = ops.topk(k=topK)(X_scores)
            bboxes = ops.batch_gather()(X_boxes, score_inds)
            keep = ops.batched_nms(iou_threshold=iou, keep_n=N)(bboxes)
            return keep, score_inds

        Y = model()
        Y[0]._attrs["is_output"] = True
        Y[0]._attrs["name"] = "output_0"
        Y[1]._attrs["is_output"] = True
        Y[1]._attrs["name"] = "output_1"

        module = compile_model(Y, target, "./tmp", test_name)

        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, 1, (N,)).cuda().half()
        y_pt = box_ops.batched_nms(boxes, scores, idxs, iou)
        y_np = y_pt.cpu().numpy()

        x = boxes.reshape(m_shape).contiguous()
        x_scores = scores.reshape((N,)).contiguous()
        inputs = {"scores": x_scores, "X": x}

        keep = torch.empty([topK]).cuda().to(torch.int64)
        score_inds = torch.empty([topK]).cuda().to(torch.int64)

        module.run_with_tensors(inputs, {"output_0": keep, "output_1": score_inds})
        keep = keep.cpu().numpy()
        score_inds = score_inds.cpu().numpy()

        index = keep.nonzero()[0]
        y = score_inds[index]
        np.testing.assert_allclose(y_np, y, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
