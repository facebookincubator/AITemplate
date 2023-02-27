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
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import parameterized


@unittest.skipIf(
    detect_target().name() == "rocm", "masked_select is not implemented for ROCm"
)
class maskedSelectTestCase(unittest.TestCase):
    def _test_masked_select(
        self,
        batch_size=1,
        shape=(2, 6),
        test_name="masked_select",
        copy_op=False,
        dtype="float16",
        zero_mask=False,
        benchmark=False,
    ):
        X1 = Tensor(
            shape=shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        X2 = Tensor(
            shape=shape,
            dtype="bool",
            name="mask",
            is_input=True,
        )
        X4_op = ops.masked_select()
        if copy_op:
            X4_op = ops.masked_select(**X4_op._get_op_attributes())

        X4 = X4_op(X1, X2)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output_values"

        target = detect_target()
        module = compile_model([X4], target, "./tmp", test_name)
        x = get_random_torch_tensor(shape, dtype=dtype)
        if zero_mask:
            mask = torch.zeros_like(x)
        else:
            mask = get_random_torch_tensor(shape, dtype="float16") > 0
        y_pt = torch.masked_select(x, mask)
        y = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
        y_ait = module.run_with_tensors([x, mask], [y])["output_values"]
        # y_ait contains the correct result. It points to the same memory blob as y, but has the correct shape
        self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-10, rtol=0))
        # y retained the original shape (x.numel(),), so needs to be cut before comparison
        self.assertTrue(torch.allclose(y_pt, y[: y_ait.shape[0]], atol=1e-10, rtol=0))

        if benchmark:
            print(f"Benchmarking with shape={shape}, dtype={dtype}")
            # Warm up.
            for _ in range(5):
                module.run_with_tensors([x, mask], [y])
            # Benchmark.
            num_benchmark_iter = 1000

            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                [x, mask], [y], count=num_benchmark_iter
            )
            print(f"AITemplate time: {time_per_iter_ms:.2f}ms")

            func = torch.masked_select
            args = (x, mask)
            # Warm up.
            for _ in range(5):
                func(*args)
            # Benchmark.
            torch_time_per_iter_ms = benchmark_torch_function(
                num_benchmark_iter, func, *args
            )
            print(f"PyTorch time: {torch_time_per_iter_ms:.2f}ms")

            print(f"Speedup: {torch_time_per_iter_ms / time_per_iter_ms:.2f}x")

    @parameterized.expand(
        [
            [(2, 6), False],
            [(20, 6), False],
            [(300, 80), False],
            # Uncomment to benchmark
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
    def test_fp16(self, shape, benchmark):
        self._test_masked_select(
            shape=shape,
            test_name="masked_select_fp16",
            dtype="float16",
            benchmark=benchmark,
        )
        if not benchmark:
            self._test_masked_select(
                shape=shape,
                test_name="masked_select_fp16_copy_op",
                copy_op=True,
                dtype="float16",
                benchmark=benchmark,
            )

    @unittest.skipIf(detect_target().name() == "rocm", "float32 not supported in ROCm")
    @parameterized.expand(
        [
            [(2, 6), False],
            [(20, 6), False],
            [(300, 80), False],
            # Uncomment to benchmark
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
        self._test_masked_select(
            shape=shape,
            test_name="masked_select_fp32",
            dtype="float32",
            benchmark=benchmark,
        )
        if not benchmark:
            self._test_masked_select(
                shape=shape,
                test_name="masked_select_fp32_copy_op",
                copy_op=True,
                dtype="float32",
                benchmark=benchmark,
            )

    def test_input_dynamic_shape(
        self,
        batch_size=1,
        shape=(2, 6),
        test_name="masked_select_dynamic",
        dtype="float16",
        benchmark=False,
    ):
        """
        Check that dynamic input shape is handled correctly.
        """
        dyn_shape = (IntVar(values=(1, 10)), IntVar(values=(1, 10)))
        X1 = Tensor(
            shape=dyn_shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        X2 = Tensor(
            shape=dyn_shape,
            dtype="bool",
            name="mask",
            is_input=True,
        )
        X4_op = ops.masked_select()
        X4 = X4_op(X1, X2)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output_values"

        target = detect_target()
        module = compile_model([X4], target, "./tmp", test_name)

        x = get_random_torch_tensor(shape, dtype=dtype)
        mask = get_random_torch_tensor(shape, dtype="float16") > 0
        y_pt = torch.masked_select(x, mask)
        y = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
        y_ait = module.run_with_tensors([x, mask], [y])["output_values"]
        # y_ait contains the correct result. It points to the same memory blob as y, but has the correct shape
        self.assertTrue(torch.allclose(y_pt, y_ait, atol=1e-10, rtol=0))
        # y retained the original shape (x.numel(),), so needs to be cut before comparison
        self.assertTrue(torch.allclose(y_pt, y[: y_ait.shape[0]], atol=1e-10, rtol=0))

    def test_empty_output(self, shape=(2, 6)):
        """
        The case when the mask is zero and the output is an empty tensor.
        """
        self._test_masked_select(
            shape=shape,
            test_name="masked_select_zero_mask",
        )


if __name__ == "__main__":
    torch.manual_seed(1024)
    unittest.main()
