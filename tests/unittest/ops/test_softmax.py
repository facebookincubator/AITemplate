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
Unittests for LayerNorm Operator.
"""
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import string_to_torch_dtype


class SoftmaxTestCase(unittest.TestCase):
    def _test_softmax(
        self,
        batch_sizes=(1, 1024),
        input_shapes=(6,),
        dim=-1,
        dtype="float16",
        testname="softmax",
    ):
        torch_dtype = string_to_torch_dtype(dtype)
        X = Tensor(
            shape=[IntVar(name="input_batch", values=list(batch_sizes)), *input_shapes],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Y = ops.softmax()(X, dim)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(Y, target, "./tmp", testname)

        for batch_size in batch_sizes:
            x_pt = torch.randn(batch_size, *input_shapes, dtype=torch_dtype).cuda()
            y_pt = torch.nn.functional.softmax(x_pt, dim=dim)

            y = torch.empty([batch_size, *input_shapes], dtype=torch_dtype).cuda()
            module.run_with_tensors([x_pt], [y])
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_softmax(self):
        self._test_softmax(testname="softmax_0")
        self._test_softmax(dim=1, testname="softmax_1")
        self._test_softmax((1, 13), (7,), testname="softmax_2")
        self._test_softmax((10, 1025), (16,), testname="softmax_3")
        self._test_softmax((1, 17), (9, 8), testname="softmax_4")
        self._test_softmax((2, 64), (9, 1, 6), testname="softmax_5")
        self._test_softmax((1, 4096), (33,), testname="softmax_6")
        self._test_softmax((2, 21), (34,), testname="softmax_7")
        self._test_softmax((2, 17), (36,), testname="softmax_8")
        self._test_softmax((1, 64), (128,), testname="softmax_9")
        self._test_softmax((2, 31), (513,), testname="softmax_10")

    def test_softmax_fp32(self):
        self._test_softmax(dtype="float32", testname="softmax_fp32_0")
        self._test_softmax(dim=1, dtype="float32", testname="softmax_fp32_1")
        self._test_softmax((1, 13), (7,), dtype="float32", testname="softmax_fp32_2")
        self._test_softmax(
            (10, 1025), (16,), dtype="float32", testname="softmax_fp32_3"
        )
        self._test_softmax((1, 17), (9, 8), dtype="float32", testname="softmax_fp32_4")
        self._test_softmax(
            (2, 64), (9, 1, 6), dtype="float32", testname="softmax_fp32_5"
        )
        self._test_softmax((1, 4096), (33,), dtype="float32", testname="softmax_fp32_6")
        self._test_softmax((2, 21), (34,), dtype="float32", testname="softmax_fp32_7")
        self._test_softmax((2, 17), (36,), dtype="float32", testname="softmax_fp32_8")
        self._test_softmax((1, 64), (128,), dtype="float32", testname="softmax_fp32_9")
        self._test_softmax((2, 31), (513,), dtype="float32", testname="softmax_fp32_10")


if __name__ == "__main__":
    unittest.main()
