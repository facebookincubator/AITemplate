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
import itertools
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMRcrN1TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_count = 0

    def _test_rcr_n1(
        self,
        Bs,
        Ms,
        N,
        K,
        use_fp16_acc,
        test_name,
        atol=1e-1,
        rtol=1e-1,
        dtype="float16",
    ):
        target = detect_target(use_fp16_acc=use_fp16_acc)
        BDim = shape_utils.gen_int_var_min_max(Bs, name="batch")
        MDim = shape_utils.gen_int_var_min_max(Ms, name="m")
        X = Tensor(
            shape=[BDim, MDim, IntImm(K)],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[BDim, IntImm(N), IntImm(K)],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.bmm_rcr_n1()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"bmm_rcr_n1_{use_fp16_acc}_{test_name}_{self.test_count}",
        )
        for B, M in itertools.product(Bs, Ms):
            logging.info(f"Testing {B=} {M=}")
            X_pt = get_random_torch_tensor((B, M, K), dtype)
            W_pt = get_random_torch_tensor((B, N, K), dtype)

            Y_pt = torch.bmm(X_pt, torch.transpose(W_pt, 2, 1))

            y = torch.empty_like(Y_pt)
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])

            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                torch.testing.assert_close(Y_pt, y, atol=atol, rtol=rtol)
        self.test_count += 1

    def test_rcr_n1(self):
        self._test_rcr_n1([1], [1000000], 1, 32, True, "static")
        self._test_rcr_n1([1], [1000000], 1, 32, False, "static")

        self._test_rcr_n1([1, 63, 96], [1024], 1, 32, True, "dynamic_batch")
        self._test_rcr_n1([1, 63, 96], [1024], 1, 32, False, "dynamic_batch")

        self._test_rcr_n1([1], [1, 1000, 100000], 1, 32, True, "dynamic_m")
        self._test_rcr_n1([1], [1, 1000, 100000], 1, 32, False, "dynamic_m")

        self._test_rcr_n1([1, 16], [1, 1024], 1, 32, True, "dynamic_batch_dynamic_m")
        self._test_rcr_n1([1, 16], [1, 1024], 1, 32, False, "dynamic_batch_dynamic_m")

        self._test_rcr_n1([1, 5, 8], [100], 1, 7, True, "static")
        self._test_rcr_n1([1, 5, 8], [100], 1, 123, False, "static")

        self._test_rcr_n1([1], [100], 1, 0, False, "zero_k")
        self._test_rcr_n1([1], [0], 1, 3, False, "zero_m")

    def test_bmm_rcr_n1_float32(self):
        self._test_rcr_n1(
            [1], [1000000], 1, 32, True, "static_float32", dtype="float32"
        )
        self._test_rcr_n1(
            [1], [1000000], 1, 32, False, "static_float32", dtype="float32"
        )
        self._test_rcr_n1(
            [1, 5, 8], [100], 1, 7, True, "static_float32", dtype="float32"
        )
        self._test_rcr_n1(
            [1, 5, 8], [100], 1, 123, False, "static_float32", dtype="float32"
        )

    @unittest.skipIf(
        int(detect_target()._arch) < 80, "bf16 is supported with CUDA sm80+"
    )
    def test_bmm_rcr_n1_bfloat16(self):
        self._test_rcr_n1(
            [1],
            [1000000],
            1,
            32,
            True,
            "static_bfloat16",
            atol=2e-1,
            rtol=2e-1,
            dtype="bfloat16",
        )
        self._test_rcr_n1(
            [1], [1000000], 1, 32, False, "static_bfloat16", dtype="bfloat16"
        )
        self._test_rcr_n1(
            [1, 5, 8], [100], 1, 7, True, "static_bfloat16", dtype="bfloat16"
        )
        self._test_rcr_n1(
            [1, 5, 8], [100], 1, 123, False, "static_bfloat16", dtype="bfloat16"
        )


if __name__ == "__main__":
    unittest.main()
