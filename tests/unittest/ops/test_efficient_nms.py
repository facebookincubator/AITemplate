#  Copyright (c) Meta Platform, Inc. and its affiliates"""
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
import os
import shutil
import unittest
from math import log
from unittest import skipIf

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.torch_utils import string_to_torch_dtype

try:
    from torchvision.ops import boxes as box_ops

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


def random_boxes(num_boxes, max_coord=100):
    boxes = torch.rand(num_boxes, 4) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nonempty(box, threshold=0.0):
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths < threshold) | (heights < threshold)
    return keep


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def create_tensors(N, dtype="float16"):
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
        dtype=dtype,
    )
    return dets[:N, :4], dets[:N, -1]


def op_gflop(bz, N, max_out):
    proposal_num = bz * N
    flop = proposal_num * log(proposal_num) + max_out * proposal_num * 16
    return flop / pow(10, 9)


@skipIfNoTorchVision
class nmsTestCase(unittest.TestCase):
    def _create_tensors(self, N, rand=False, dtype="float16"):
        if rand:
            boxes = random_boxes(N, 200)
            scores = torch.rand(N)
            return (
                boxes.numpy().astype(dtype),
                scores.numpy().astype(dtype),
            )
        else:
            boxes, scores = create_tensors(N, dtype=dtype)
            return boxes, scores

    def _test_nms(
        self,
        batch_size=1,
        N=100,
        num_classes=1,
        preNmsTop=100,
        nmsMaxOut=100,
        iouThreshold=0.5,
        confidence=0.5,
        minBoxSize=16,
        rand_box=False,
        bench_pt=False,
        rebuild=True,
        test_name="efficient_nms",
        benchmark_shapes=False,
        copy_op=False,
        dtype="float16",
    ):
        X1 = Tensor(
            shape=[batch_size, N, num_classes, 4],
            dtype=dtype,
            name="boxes",
            is_input=True,
        )

        X2 = Tensor(
            shape=[batch_size, N, num_classes],
            dtype=dtype,
            name="scores",
            is_input=True,
        )

        OP = ops.efficient_nms(
            preNmsTop=preNmsTop,
            nmsMaxOut=nmsMaxOut,
            iouThreshold=iouThreshold,
            minBoxSize=minBoxSize,
        )
        if copy_op:
            OP = ops.efficient_nms(**OP._get_op_attributes())
        Y = OP(X1, X2)
        mark_output(Y)

        torch_dtype = string_to_torch_dtype(dtype)
        boxes, scores = self._create_tensors(N, rand=rand_box, dtype=dtype)
        idxs = torch.randint(0, num_classes, (N,)).cuda().to(dtype=torch_dtype)
        boxes_pt = torch.tensor(boxes).cuda().to(dtype=torch_dtype)
        kept = nonempty(boxes_pt, threshold=minBoxSize)
        score_pt = torch.tensor(scores).cuda().to(dtype=torch_dtype)
        score_pt[kept] = -1

        if bench_pt:
            func = box_ops.batched_nms
            args = (boxes_pt, score_pt, idxs, iouThreshold)
            batch_size = 1
            duration = benchmark_torch_function(100, func, *args)
            print(
                f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
            )

        keep = box_ops.batched_nms(boxes_pt, score_pt, idxs, iouThreshold)

        if keep.shape[0] >= nmsMaxOut:
            keep = keep[:nmsMaxOut]
            ref_box = boxes_pt[keep].cpu()
        else:
            ref_box = torch.zeros(nmsMaxOut, 4)
            ref_box[
                : keep.shape[0],
            ] = boxes_pt[keep].cpu()
        ref_box = ref_box.cuda().to(dtype=torch_dtype)

        x = boxes.reshape((1, N, 1, 4)).copy()
        x_scores = scores.reshape((1, N, 1)).copy()

        x = np.repeat(x, repeats=num_classes, axis=2)
        x_scores = np.repeat(x_scores, repeats=num_classes, axis=2)
        x = np.repeat(x, repeats=batch_size, axis=0)
        x_scores = np.repeat(x_scores, repeats=batch_size, axis=0)

        rebuild = 1
        target = detect_target()
        if rebuild:
            try:
                shutil.rmtree("./tmp/" + str(test_name))
            except FileNotFoundError:
                pass
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        x_reshaped = torch.from_numpy(x.reshape(batch_size, N, num_classes, 4)).cuda()
        scores_reshaped = torch.from_numpy(
            x_scores.reshape(batch_size, N, num_classes)
        ).cuda()
        inputs = {"boxes": x_reshaped, "scores": scores_reshaped}

        y0 = torch.empty([batch_size, 1]).cuda().to(torch.int64)
        y1 = torch.empty([batch_size, nmsMaxOut, 4]).cuda().to(dtype=torch_dtype)
        y2 = torch.empty([batch_size, nmsMaxOut]).cuda().to(dtype=torch_dtype)
        y3 = torch.empty([batch_size, nmsMaxOut]).cuda().to(dtype=torch.int64)
        outputs = {"output_0": y0, "output_1": y1, "output_2": y2, "output_3": y3}
        module.run_with_tensors(inputs, outputs)

        if benchmark_shapes:
            module.benchmark_with_tensors(inputs, outputs)
            gflop = op_gflop(batch_size, N, nmsMaxOut)
            print(
                f"NMS op gflop [batch size={batch_size}, N={N}, nmsMaxOut={nmsMaxOut}]: {gflop}"
            )
            return

        module.run_with_tensors(inputs, outputs)
        if batch_size > 1 and num_classes > 1:
            idx1, idx2 = 0, -1
            for y in [y0, y1, y2, y3]:
                self.assertTrue(
                    torch.allclose(y[idx1, :], y[idx2, :], atol=1e-2, rtol=1e-2)
                )
        else:
            self.assertTrue(torch.allclose(y1[0, :], ref_box, atol=1e-2, rtol=1e-2))

    def test_nms_fp16(self):
        # self._test_nms(
        #     N=15000,
        #     preNmsTop=6000,
        #     nmsMaxOut=1000,
        #     iouThreshold=0.7,
        #     minBoxSize=0,
        #     batch_size=2,
        #     rand_box=True,
        #     test_name="nms1",
        # )

        """
        self._test_nms(
            N=30,
            preNmsTop=30,
            nmsMaxOut=10,
            iouThreshold=0.5,
            minBoxSize=0,
            batch_size=1,
            num_classes=1,
            rand_box=False,
            test_name="nms1",
        )
        """
        self._test_nms(
            N=30,
            preNmsTop=30,
            nmsMaxOut=10,
            iouThreshold=0.5,
            minBoxSize=0,
            batch_size=2,
            num_classes=4,
            rand_box=False,
            test_name="nms2_fp16",
            dtype="float16",
        )
        self._test_nms(
            N=30,
            preNmsTop=30,
            nmsMaxOut=10,
            iouThreshold=0.5,
            minBoxSize=0,
            batch_size=2,
            num_classes=4,
            rand_box=False,
            test_name="nms2_copy_op_fp16",
            copy_op=True,
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "float32 not supported in ROCm")
    def test_nms_fp32(self):
        self._test_nms(
            N=30,
            preNmsTop=30,
            nmsMaxOut=10,
            iouThreshold=0.5,
            minBoxSize=0,
            batch_size=2,
            num_classes=4,
            rand_box=False,
            test_name="nms2_fp32",
            dtype="float32",
        )
        self._test_nms(
            N=30,
            preNmsTop=30,
            nmsMaxOut=10,
            iouThreshold=0.5,
            minBoxSize=0,
            batch_size=2,
            num_classes=4,
            rand_box=False,
            test_name="nms2_copy_op_fp32",
            copy_op=True,
            dtype="float32",
        )

    # !!! SKIPPED TESTS BELOW !!!
    # manually enable for benchmarking

    # def test_nms_benchmark_shapes(self):
    #     self._test_nms(
    #         N=3350,
    #         preNmsTop=2000,
    #         nmsMaxOut=100,
    #         iouThreshold=0.5,
    #         minBoxSize=0,
    #         batch_size=16,
    #         num_classes=1,
    #         rand_box=True,
    #         test_name="nms_fcos_shape",
    #         benchmark_shapes=True,
    #     )

    #     for bz in (1, 4, 16):
    #         for N in (6000, 12000, 20000, 60000):
    #             for maxout in (100, 300, 1000):
    #                 self._test_nms(
    #                     N=N,
    #                     preNmsTop=6000,
    #                     nmsMaxOut=maxout,
    #                     iouThreshold=0.5,
    #                     minBoxSize=0,
    #                     batch_size=bz,
    #                     num_classes=1,
    #                     rand_box=True,
    #                     test_name="nms_" + str(bz) + "_" + str(N) + "_" + str(maxout),
    #                     benchmark_shapes=True,
    #                 )


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
