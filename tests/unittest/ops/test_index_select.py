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
Unittests for masked_select Operator.
"""
import logging
import random
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import parameterized

logger = logging.getLogger(__name__)


@unittest.skipIf(
    detect_target().name() == "rocm", "masked_select is not implemented for ROCm"
)
class IndexSelectTest(unittest.TestCase):
    @staticmethod
    def _get_output_shape(shape, dim_idx, dim_idx_len):
        ret = []
        for idx, dim in enumerate(shape):
            if idx == dim_idx:
                ret.append(dim_idx_len)
                continue
            ret.append(dim)
        return ret

    def _test_index_select(
        self,
        shape=(2, 2),
        x_shape=None,
        dim_idxs_shape=None,
        dim_idx=1,
        dim_idx_len=1,
        test_name="index_select",
        dtype="float16",
        benchmark=False,
        dim_idxs=None,
    ):

        X1 = Tensor(
            shape=shape if x_shape is None else x_shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        X2 = Tensor(
            shape=(dim_idx_len,) if dim_idxs_shape is None else dim_idxs_shape,
            dtype="int64",
            name="dim_idxs",
            is_input=True,
        )
        X4_op = ops.index_select(dim_idx)
        X4 = X4_op(X1, X2)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output_values"

        target = detect_target()
        module = compile_model([X4], target, "./tmp", test_name)
        x = get_random_torch_tensor(shape, dtype=dtype)
        y = torch.empty(
            IndexSelectTest._get_output_shape(shape, dim_idx, dim_idx_len),
            dtype=x.dtype,
            device=x.device,
        )

        if dim_idxs is None:
            dim_idxs = torch.arange(end=dim_idx_len, dtype=torch.int64, device=x.device)

        y_ait = module.run_with_tensors([x, dim_idxs], [y])["output_values"]
        y_pt = torch.index_select(x, dim_idx, dim_idxs)
        self.assertTrue(torch.equal(y_pt, y_ait))

        if benchmark:
            print(
                f"Benchmarking with shape={shape}, dim_idx={dim_idx}, dim_idx_len={dim_idx_len}, dtype={dtype}"
            )
            # Warm up.
            for _ in range(5):
                module.run_with_tensors([x, dim_idxs], [y])
            # Benchmark.
            num_benchmark_iter = 1000

            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                [x, dim_idxs], [y], count=num_benchmark_iter
            )

            print(f"AITemplate time: {time_per_iter_ms:.4f}ms")

            func = torch.index_select
            args = (x, dim_idx, dim_idxs)
            # Warm up.
            for _ in range(5):
                func(*args)
            # Benchmark.
            torch_time_per_iter_ms = benchmark_torch_function(
                num_benchmark_iter, func, *args
            )
            print(f"PyTorch time: {torch_time_per_iter_ms:.4f}ms")

            print(f"Speedup: {torch_time_per_iter_ms / time_per_iter_ms:.6f}x")

    @unittest.skipIf(detect_target().name() == "rocm", "float32 not supported in ROCm")
    @parameterized.expand(
        [
            [
                (IntVar(values=[1, 6]), IntVar(values=[1, 6])),
                (IntVar(values=(0, 2)),),
                (2, 2),
                False,
            ],
            [
                (IntVar(values=[1, 2048]), 1024, 7),
                (IntVar(values=[1, 512]),),
                (2048, 1024, 7),
                False,  # change for benchmark
                2,
                7,
            ],
            [
                (IntVar(values=[1, 2048]), 1024, 7),
                (IntVar(values=[1, 512]),),
                (2048, 1024, 7),
                False,
                1,
                512,
            ],
            [
                (IntVar(values=[1, 2048]), 1024, 7),
                (IntVar(values=[1, 2048]),),
                (2048, 1024, 7),
                False,
                0,
                2048,
            ],
        ]
    )
    def test_dynamic_shape(
        self,
        x_shape=None,
        dim_idxs_shape=None,
        shape=(2, 2),
        benchmark=False,
        dim_idx=1,
        dim_idx_len=1,
        test_name="dynamic_index_select",
        dtype="float16",
    ):
        self._test_index_select(
            shape,
            x_shape,
            dim_idxs_shape,
            dim_idx,
            dim_idx_len,
            test_name,
            dtype,
            benchmark,
        )

    def test_repeated_and_out_of_order(self):
        self._test_index_select(
            shape=(5, 4, 3, 2),
            dim_idx=1,
            dim_idx_len=10,
            test_name="index_select_repeat",
            dtype="float16",
            dim_idxs=torch.tensor(
                [3, 2, 0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.int64, device="cuda"
            ),
        )

    def test_negative_dim(self):
        for dim_idx in range(1, 5):
            self._test_index_select(
                shape=(5, 4, 3, 2),
                dim_idx=-dim_idx,
                dim_idx_len=1,
                test_name="index_select_negative_idx",
                dtype="float16",
            )

    @unittest.skipIf(detect_target().name() == "rocm", "float32 not supported in ROCm")
    @parameterized.expand(
        [
            [(5, 4, 3, 2), False],
            # [(2, 6), False],
            # [(20, 6), False],
            # [(300, 80), False],
            # Uncomment to benchmark
            # [(5, 4, 3, 2), False],
            # [(2, 6), False],
            # [(20, 6), False],
            # [(300, 80), True],
            # [(1024, 128, 256), True],
            # [(1024, 1024, 100), True],
            # [(1, 1), True],
            # [(10, 1), True],
            # [(100, 1), True],
            # [(1000, 1), True],
            # [(10000, 1), True],
            # [(100000, 1), True],
            # [(1000000, 1), True],
            # [(10000000, 1), True],
            # [(100000000, 1), True],
            # [(10000, 10000), True],
            # [(10, 10, 10, 10, 10, 10, 10, 10), True],
        ]
    )
    def test_fp32(self, shape, benchmark):
        torch.manual_seed(1024)
        random.seed(1024)
        for idx, _ in enumerate(shape):
            for dim_idx_len in [1, int(shape[idx] / 2), shape[idx]]:
                self._test_index_select(
                    shape=shape,
                    dim_idx=idx,
                    dim_idx_len=dim_idx_len if dim_idx_len > 0 else 1,
                    test_name="index_select_fp32",
                    dtype="float32",
                    benchmark=benchmark,
                )

    @parameterized.expand(
        [
            [(5, 4, 3, 2), False],
            # [(2, 6), False],
            # [(20, 6), False],
            # [(300, 80), False],
            # Uncomment to benchmark
            # [(5, 4, 3, 2), True],
            # [(2, 6), True],
            # [(20, 6), True],
            # [(300, 80), True],
            # [(1024, 128, 256), True],
            # [(1024, 1024, 100), True],
            # [(1, 1), True],
            # [(10, 1), True],
            # [(100, 1), True],
            # [(1000, 1), True],
            # [(10000, 1), True], #revisit
            # [(100000, 1), True],
            # [(1000000, 1), True],
            # [(10000000, 1), True],
            # [(100000000, 1), True],
            # [(10000, 10000), True],
            # [(10, 10, 10, 10, 10, 10, 10, 10), True],
        ]
    )
    def test_fp16(self, shape, benchmark=False):
        torch.manual_seed(1024)
        random.seed(1024)
        for idx, _ in enumerate(shape):
            for dim_idx_len in [1, int(shape[idx] / 2), shape[idx]]:
                self._test_index_select(
                    shape=shape,
                    dim_idx=idx,
                    dim_idx_len=dim_idx_len if dim_idx_len > 0 else 1,
                    test_name="index_select_fp16",
                    dtype="float16",
                    benchmark=benchmark,
                )


if __name__ == "__main__":
    torch.manual_seed(1024)
    random.seed(1024)
    unittest.main()
