# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for topk Operator.
"""
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


class topkTestCase(unittest.TestCase):
    def _create_tensors(self, shape):
        N = np.prod(shape)
        scores = torch.randperm(N) / N
        return scores.reshape(shape).cuda().half()

    def _test_topk(
        self, batch_size=1, shape=(2, 500), dim=0, topK=100, test_name="topk"
    ):

        o_shape = list(shape)
        o_shape[-1] = topK

        X1 = Tensor(
            shape=shape,
            dtype="float16",
            name="X",
            is_input=True,
        )
        X4 = ops.topk(k=topK)(X1)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = gen_execution_module(X4, target, "./tmp", test_name)

        scores = self._create_tensors(shape)
        (values, y_pt) = torch.topk(scores, k=topK, dim=dim)

        x = scores.reshape(shape).contiguous()
        y = torch.empty(o_shape).cuda().to(torch.int64)
        module.RunWithTensors([x], [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_topk_heap(self):
        self._test_topk(shape=(2000,), topK=100, test_name="topk_heap")
        self._test_topk(shape=(4, 500), topK=100, dim=1, test_name="topk_heap2")

    def test_topk_sort(self):
        self._test_topk(shape=(2000,), topK=300, test_name="topk_sort")
        self._test_topk(shape=(4, 500), topK=200, dim=1, test_name="topk_sort2")


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
