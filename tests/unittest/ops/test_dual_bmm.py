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
import logging
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(detect_target()._arch == "75", "DualGemm not supported on sm75.")
class DUALBMMTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_dual_bmm_rrr_div(
        self,
        B=256,
        M=256,
        N=512,
        K=512,
        broadcast_b1=False,
        benchmark=False,
        use_fp16_acc=False,
        test_name="dual_bmm",
        dtype="float16",
    ):
        B1_shape = [B, K, 1] if broadcast_b1 else [B, K, N]
        target = detect_target(use_fp16_acc=use_fp16_acc)
        X = Tensor(
            shape=[B, M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        B0 = Tensor(
            shape=[B, K, N],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B1 = Tensor(
            shape=B1_shape,
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.dual_bmm_rrr_div()
        Y = OP(X, B0, B1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        X_pt = get_random_torch_tensor([B, M, K], dtype=dtype) + 1.0
        B0_pt = get_random_torch_tensor([B, K, N], dtype=dtype) + 1.0
        B1_pt = get_random_torch_tensor(B1_shape, dtype=dtype) + 1.0

        def pt_func(X_pt, W_pt, B_pt):
            Y_pt1 = torch.bmm(X_pt, W_pt)
            Y_pt2 = torch.bmm(X_pt, B_pt)
            Y_pt = Y_pt1 / Y_pt2
            return Y_pt

        Y_pt = pt_func(X_pt, B0_pt, B1_pt)

        inputs = {"input_0": X_pt, "input_1": B0_pt, "input_2": B1_pt}
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])

        torch.testing.assert_close(Y_pt, y, atol=1e-2, rtol=1e-2)

        if benchmark:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark AIT
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            _LOGGER.info(f"[{M}, {N}, {K}] AIT BMMxBMM time: {time_per_iter_ms:.5f}ms")
            # Benchmark PT
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = pt_func
            args = (X_pt, B0_pt, B1_pt)
            duration = benchmark_torch_function(100, func, *args)
            _LOGGER.info(f"PT BMMxBMM Time: {duration:.5f}ms")

    def test_dual_bmm_rrr_div_fp16(self):
        self._test_dual_bmm_rrr_div(
            B=37,
            M=63,
            N=64,
            K=128,
            broadcast_b1=False,
            test_name="dual_bmm_rrr_div_fp16_1",
            dtype="float16",
        )
        self._test_dual_bmm_rrr_div(
            B=512,
            M=256,
            N=512,
            K=512,
            broadcast_b1=False,
            test_name="dual_bmm_rrr_div_fp16_2",
            dtype="float16",
        )
        self._test_dual_bmm_rrr_div(
            B=64,
            M=1024,
            N=1024,
            K=2048,
            broadcast_b1=False,
            test_name="dual_bmm_rrr_div_fp16_3",
            dtype="float16",
        )

    def test_dual_bmm_rrr_div_broadcast_b1_fp16(self):
        self._test_dual_bmm_rrr_div(
            B=37,
            M=63,
            N=64,
            K=128,
            broadcast_b1=True,
            test_name="dual_bmm_rrr_div_broadcast_b1_fp16_1",
            dtype="float16",
        )
        # self._test_dual_bmm_rrr_div(
        #     B=512,
        #     M=256,
        #     N=512,
        #     K=512,
        #     broadcast_b1=True,
        #     test_name="dual_bmm_rrr_div_broadcast_b1_fp16_2",
        #     dtype="float16",
        # )
        # self._test_dual_bmm_rrr_div(
        #     B=64,
        #     M=1024,
        #     N=1024,
        #     K=2048,
        #     broadcast_b1=True,
        #     test_name="dual_bmm_rrr_div_broadcast_b1_fp16_3",
        #     dtype="float16",
        # )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_dual_bmm_rrr_div_fp32(self):
        self._test_dual_bmm_rrr_div(
            B=37,
            M=63,
            N=64,
            K=128,
            broadcast_b1=False,
            test_name="dual_bmm_rrr_div_fp32_1",
            dtype="float32",
        )
        # self._test_dual_bmm_rrr_div(
        #     B=512,
        #     M=256,
        #     N=512,
        #     K=512,
        #     broadcast_b1=False,
        #     test_name="dual_bmm_rrr_div_fp32_2",
        #     dtype="float32",
        # )
        # self._test_dual_bmm_rrr_div(
        #     B=64,
        #     M=1024,
        #     N=1024,
        #     K=2048,
        #     broadcast_b1=False,
        #     test_name="dual_bmm_rrr_div_fp32_3",
        #     dtype="float32",
        # )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_dual_bmm_rrr_div_broadcast_b1_fp32(self):
        self._test_dual_bmm_rrr_div(
            B=37,
            M=63,
            N=64,
            K=128,
            broadcast_b1=True,
            test_name="dual_bmm_rrr_div_broadcast_b1_fp32_1",
            dtype="float32",
        )
        # self._test_dual_bmm_rrr_div(
        #     B=512,
        #     M=256,
        #     N=512,
        #     K=512,
        #     broadcast_b1=True,
        #     test_name="dual_bmm_rrr_div_broadcast_b1_fp32_2",
        #     dtype="float32",
        # )
        # self._test_dual_bmm_rrr_div(
        #     B=64,
        #     M=1024,
        #     N=1024,
        #     K=2048,
        #     broadcast_b1=True,
        #     test_name="dual_bmm_rrr_div_broadcast_b1_fp32_3",
        #     dtype="float32",
        # )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
