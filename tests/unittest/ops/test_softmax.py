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
import json
import math
import tempfile
import unittest
from collections import namedtuple
from statistics import mean

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.profile import profile_callable
from aitemplate.testing.test_utils import filter_test_cases_by_params, TestEnv
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import parameterized


class SoftmaxTestCase(unittest.TestCase):
    def _build_model(
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

        X = Tensor(
            shape=[IntVar(name="input_batch", values=list(batch_sizes)), *input_shapes],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Y = ops.softmax()(X, dim)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "Y"

        return compile_model(Y, target, "./tmp", testname)

    def _test_softmax(
        self,
        batch_sizes=(1, 1024),
        input_shapes=(6,),
        dim=-1,
        dtype="float16",
        testname="softmax",
    ):
        module = self._build_model(batch_sizes, input_shapes, dim, dtype, testname)
        torch_dtype = string_to_torch_dtype(dtype)

        for batch_size in batch_sizes:
            x_pt = torch.randn(batch_size, *input_shapes, dtype=torch_dtype).cuda()
            y_pt = torch.nn.functional.softmax(x_pt, dim=dim)

            y = torch.empty([batch_size, *input_shapes], dtype=torch_dtype).cuda()
            module.run_with_tensors([x_pt], [y])
            torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [
                    ("dim_1_fp16", "float16", (1, 1024), (6,), 1),
                    ("tail_shapes_all_1_fp16", "float16", (1, 2), (6, 1, 1), 1),
                    ("tail_shapes_not_all_1_fp16", "float16", (1, 2), (6, 1, 2), 1),
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
                ],
                TestEnv.CUDA_SM80: [
                    ("dim_1_bf16", "bfloat16", (1, 2), (6,), 1),
                    ("tail_shapes_all_1_bf16", "bfloat16", (1, 2), (6, 1, 1), 1),
                    ("tail_shapes_not_all_1_bf16", "bfloat16", (1, 2), (6, 1, 2), 1),
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
                ],
            }
        )
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

    def _test_benchmark_softmax(self):
        dtype = "float16"
        torch_dtype = string_to_torch_dtype(dtype)
        BenchResult = namedtuple(
            "BenchResult", ["dim", "batch_size", "permute_ms", "softmax_ms"]
        )
        results = []
        shape = (260, 4)
        batch_sizes = [2**p for p in range(0, 16)]
        for reduction_dim in [-1, -2]:
            module = self._build_model(
                batch_sizes,
                shape,
                reduction_dim,
                dtype,
                f"bench_softmax_{abs(reduction_dim)}",
            )

            for batch_size in batch_sizes:
                x_pt = torch.ones(batch_size, *shape, dtype=torch_dtype).cuda()
                y_pt = torch.empty([batch_size, *shape], dtype=torch_dtype).cuda()
                with tempfile.NamedTemporaryFile("r") as f:
                    module.profile_with_tensors(
                        inputs={"X": x_pt},
                        outputs={"Y": y_pt},
                        num_iters=1000,
                        filename=f.name,
                    )
                    profiling_data = json.loads(f.read())

                    permute_ms = 0
                    softmax_ms = 0
                    for func_name, record in profiling_data.items():
                        if func_name.startswith("permute"):
                            permute_ms += record["ms_per_iter"]
                        elif func_name.startswith("softmax"):
                            softmax_ms += record["ms_per_iter"]
                    results.append(
                        BenchResult(reduction_dim, batch_size, permute_ms, softmax_ms)
                    )

        for r in results:
            items = r.batch_size * math.prod(shape)
            runtime_ms = r.permute_ms + r.softmax_ms
            print(
                f"{r.dim=}, {items=}, {r.permute_ms=}, {r.softmax_ms=}, {runtime_ms=}"
            )

    def _test_benchmark_pytorch_softmax(self):
        batch_sizes = [2**p for p in range(0, 16)]
        shape = (260, 4)
        dtype = "float16"
        torch_dtype = string_to_torch_dtype(dtype)
        BenchResult = namedtuple("BenchResult", ["dim", "batch_size", "runtime_ms"])
        cache_flush_slab = torch.empty(
            size=[40, 1024, 1024],  # A100 L2 cache size
            dtype=torch.float16,
        ).cuda()

        results = []
        for reduction_dim in [-1, -2]:
            for batch_size in batch_sizes:
                x_pt = torch.ones(batch_size, *shape, dtype=torch_dtype).cuda()
                _, wall_times = profile_callable(
                    lambda: torch.nn.functional.softmax(x_pt, dim=reduction_dim),
                    cache_flush_slab,
                    n_iter=1000,
                )
                results.append(
                    BenchResult(reduction_dim, batch_size, mean(wall_times) / 1000.0)
                )

        for r in results:
            items = r.batch_size * math.prod(shape)
            print(f"{r.dim=}, {items=}, {r.runtime_ms=}")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
