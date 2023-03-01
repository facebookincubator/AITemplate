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
from parameterized import parameterized


class SoftmaxTestCase(unittest.TestCase):
    def _test_softmax(
        self,
        batch_sizes=(1, 1024),
        input_shapes=(6,),
        dim=-1,
        dtype="float16",
        testname="softmax",
    ):
        target = detect_target()
        if target.name() == "rocm" and dtype != "float16":
            self.skipTest(f"Rocm doesn't support {dtype}")
        if target.name() == "cuda" and dtype == "bfloat16" and int(target._arch) < 80:
            self.skipTest(f"CUDA SM{target._arch} doesn't support {dtype}")
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

        module = compile_model(Y, target, "./tmp", testname)

        for batch_size in batch_sizes:
            x_pt = torch.randn(batch_size, *input_shapes, dtype=torch_dtype).cuda()
            y_pt = torch.nn.functional.softmax(x_pt, dim=dim)

            y = torch.empty([batch_size, *input_shapes], dtype=torch_dtype).cuda()
            module.run_with_tensors([x_pt], [y])
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        [
            ("dim_1_fp16", "float16", (1, 1024), (6,), 1),
            ("odd_small_fp16", "float16", (1, 13), (11,)),
            ("odd_mid_fp16", "float16", (1, 4096), (33,)),
            ("odd_large_fp16", "float16", (2, 31), (1409,)),
            ("k2_small_fp16", "float16", (1, 1024), (18,)),
            ("k2_mid_fp16", "float16", (2, 21), (66,)),
            ("k2_large_fp16", "float16", (2, 21), (1154,)),
            ("k4_small_fp16", "float16", (10, 1025), (124,)),
            ("k4_mid_fp16", "float16", (1, 17), (132,)),
            ("k4_large_fp16", "float16", (1, 17), (1924,)),
            ("k8_small_fp16", "float16", (10, 1025), (72,)),
            ("k8_mid_fp16", "float16", (1, 17), (264,)),
            ("k8_large_fp16", "float16", (1, 17), (3848,)),
            ("no_smem_fp16", "float16", (1, 2), (12500,)),
            ("2d", "float16", (1, 2), (100, 100)),
            ("3d", "float16", (1, 2), (24, 2, 64)),
            ("dim_1_fp32", "float32", (1, 2), (6,), 1),
            ("odd_small_fp32", "float32", (1, 2), (11,)),
            ("odd_mid_fp32", "float32", (1, 2), (33,)),
            ("odd_large_fp32", "float32", (1, 2), (1409,)),
            ("k2_small_fp32", "float32", (1, 2), (18,)),
            ("k2_mid_fp32", "float32", (1, 2), (66,)),
            ("k2_large_fp32", "float32", (1, 2), (1154,)),
            ("k4_small_fp32", "float32", (1, 2), (124,)),
            ("k4_mid_fp32", "float32", (1, 2), (132,)),
            ("k4_large_fp32", "float32", (1, 2), (1924,)),
            ("k8_small_fp32", "float32", (1, 2), (72,)),
            ("k8_mid_fp32", "float32", (1, 2), (264,)),
            ("k8_large_fp32", "float32", (1, 2), (3848,)),
            ("no_smem_fp32", "float32", (1, 2), (12500,)),
            ("dim_1_bf16", "bfloat16", (1, 2), (6,), 1),
            ("odd_small_bf16", "bfloat16", (1, 2), (11,)),
            ("odd_mid_bf16", "bfloat16", (1, 2), (33,)),
            ("odd_large_bf16", "bfloat16", (1, 2), (1409,)),
            ("k2_small_bf16", "bfloat16", (1, 2), (18,)),
            ("k2_mid_bf16", "bfloat16", (1, 2), (66,)),
            ("k2_large_bf16", "bfloat16", (1, 2), (1154,)),
            ("k4_small_bf16", "bfloat16", (1, 2), (124,)),
            ("k4_mid_bf16", "bfloat16", (1, 2), (132,)),
            ("k4_large_bf16", "bfloat16", (1, 2), (1924,)),
            ("k8_small_bf16", "bfloat16", (1, 2), (72,)),
            ("k8_mid_bf16", "bfloat16", (1, 2), (264,)),
            ("k8_large_bf16", "bfloat16", (1, 2), (3848,)),
            ("no_smem_bf16", "bfloat16", (1, 2), (12500,)),
        ]
    )
    def test_softmax(
        self,
        testname="softmax",
        dtype="float16",
        batch_sizes=(1, 1024),
        input_shapes=(6,),
        dim=-1,
    ):
        self._test_softmax(
            dtype=dtype,
            testname=f"{testname}_{dtype}",
            batch_sizes=batch_sizes,
            input_shapes=input_shapes,
            dim=dim,
        )


if __name__ == "__main__":
    unittest.main()
