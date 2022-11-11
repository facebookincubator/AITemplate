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
Unittests for topk Operator.
"""
import unittest

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class topkTestCase(unittest.TestCase):
    def _create_tensors(self, shape):
        N = np.prod(shape)
        scores = torch.randperm(N) / N
        return scores.reshape(shape).cuda().half()

    def _test_topk(
        self,
        batch_size=1,
        shape=(2, 500),
        dim=0,
        topK=100,
        test_name="topk",
        copy_op=False,
    ):

        o_shape = list(shape)
        o_shape[-1] = topK

        X1 = Tensor(
            shape=shape,
            dtype="float16",
            name="X",
            is_input=True,
        )
        OP = ops.topk(k=topK)
        if copy_op:
            OP = ops.topk(**OP._get_op_attributes())
        X4 = OP(X1)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(X4, target, "./tmp", test_name)

        scores = self._create_tensors(shape)
        (values, y_pt) = torch.topk(scores, k=topK, dim=dim)

        x = scores.reshape(shape).contiguous()
        y = torch.empty(o_shape).cuda().to(torch.int64)
        module.run_with_tensors([x], [y])
        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_topk_heap(self):
        self._test_topk(shape=(2000,), topK=100, test_name="topk_heap")
        self._test_topk(
            shape=(2000,), topK=100, test_name="topk_heap_copy_op", copy_op=True
        )
        self._test_topk(shape=(4, 500), topK=100, dim=1, test_name="topk_heap2")
        self._test_topk(
            shape=(4, 500),
            topK=100,
            dim=1,
            test_name="topk_heap2_copy_op",
            copy_op=True,
        )

    def test_topk_sort(self):
        self._test_topk(shape=(2000,), topK=300, test_name="topk_sort")
        self._test_topk(
            shape=(2000,), topK=300, test_name="topk_sort_copy_op", copy_op=True
        )
        self._test_topk(shape=(4, 500), topK=200, dim=1, test_name="topk_sort2")
        self._test_topk(
            shape=(4, 500),
            topK=200,
            dim=1,
            test_name="topk_sort2_copy_op",
            copy_op=True,
        )


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
