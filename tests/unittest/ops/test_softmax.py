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

from aitemplate.testing import detect_target

from aitemplate.testing.driver import AITTestCase
from aitemplate.testing.modules.softmax_ait import SoftmaxAIT
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import parameterized


def test_cases():
    params = [
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
    ]
    target = detect_target()
    if target.name() == "cuda" and int(target._arch) >= 80:
        params = [
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
    return params


class SoftmaxTestCase(AITTestCase):
    @parameterized.expand(test_cases)
    def test_softmax(
        self,
        testname="softmax",
        dtype="float16",
        batch_sizes=(1, 1024),
        input_shapes=(6,),
        dim=-1,
    ):
        ait_mod = SoftmaxAIT(
            dim,
            module_name=f"{testname}_{self._test_id}",
            batch_sizes=batch_sizes,
            target=self.target,
        )
        self._renew_id()

        pt_mod = torch.nn.Softmax(dim=dim)

        for batch_size in batch_sizes:
            x_pt = get_random_torch_tensor((batch_size, *input_shapes))
            y_pt = pt_mod(x_pt)
            y_ait = ait_mod(x_pt)
            torch.testing.assert_close(y_pt, y_ait, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
