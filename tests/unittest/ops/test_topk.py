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
from aitemplate.utils.torch_utils import string_to_torch_dtype


class topkTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_count = 0

    def _create_tensors(self, shape, dtype):
        N = np.prod(shape)
        scores = torch.randperm(N) / N
        return scores.reshape(shape).cuda().to(dtype=string_to_torch_dtype(dtype))

    def _test_topk(
        self,
        batch_size=1,
        shape=(2, 500),
        dim=0,
        topK=100,
        test_name="topk",
        copy_op=False,
        dtype="float16",
    ):
        o_shape = list(shape)
        o_shape[-1] = topK

        X1 = Tensor(
            shape=shape,
            dtype=dtype,
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
        module = compile_model(X4, target, "./tmp", f"{test_name}_{self.test_count}")

        scores = self._create_tensors(shape, dtype)
        (values, y_pt) = torch.topk(scores, k=topK, dim=dim)

        x = scores.reshape(shape).contiguous()
        y = torch.empty(o_shape).cuda().to(torch.int64)
        module.run_with_tensors([x], [y])
        torch.testing.assert_close(y_pt, y, atol=0, rtol=0)
        self.test_count += 1

    def test_topk_heap(self):
        self._test_topk(shape=(2000,), topK=30, test_name="topk_heap")
        self._test_topk(
            shape=(2000,), topK=40, test_name="topk_heap_copy_op", copy_op=True
        )
        self._test_topk(shape=(4, 500), topK=50, dim=1, test_name="topk_heap2")
        self._test_topk(
            shape=(4, 500),
            topK=2,
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

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported by ROCm.")
    def test_float32(self):
        self._test_topk(
            shape=(4, 500),
            topK=200,
            dim=1,
            test_name="topk_sort_f32",
            copy_op=False,
            dtype="float32",
        )
        self._test_topk(
            shape=(4, 500),
            topK=30,
            dim=1,
            test_name="topk_heap_f32",
            copy_op=False,
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
