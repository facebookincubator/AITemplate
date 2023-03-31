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
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMSoftmaxTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_gemm_rcr_softmax(
        self,
        M=16,
        K=64,
        N=24,
        dtype="float16",
        test_name="gemm_rcr_softmax",
    ):
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rcr_softmax()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        x_pt = get_random_torch_tensor([M, K], dtype)
        w_pt = get_random_torch_tensor([N, K], dtype)
        y_pt = torch.nn.functional.linear(x_pt, w_pt)
        y_pt = torch.softmax(y_pt, dim=1)

        module = compile_model(Y, detect_target(), "./tmp", test_name)

        inputs = {"input_0": x_pt, "input_1": w_pt}
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])

        torch.testing.assert_close(y, y_pt, atol=1e-2, rtol=1e-2)

        torch.testing.assert_close(
            torch.argmax(y, axis=1),
            torch.argmax(y_pt, axis=1),
            atol=1e-1,
            rtol=1e-1,
        )

    def test_gemm_rcr_softmax_float16(self):
        self._test_gemm_rcr_softmax(
            M=16,
            K=64,
            N=24,
            dtype="float16",
            test_name="gemm_rcr_softmax_fp16_1",
        )

        if not detect_target().use_dummy_profiling_results():
            # dummy workspace size (10240 bytes) is insufficient for
            # these tests: run them only locally where profiler is
            # executed and detects the necessary workspace size
            self._test_gemm_rcr_softmax(
                M=1024,
                K=512,
                N=4096,
                dtype="float16",
                test_name="gemm_rcr_softmax_fp16_2",
            )
            self._test_gemm_rcr_softmax(
                M=2048,
                K=1024,
                N=4096,
                dtype="float16",
                test_name="gemm_rcr_softmax_fp16_3",
            )

    def test_gemm_rcr_softmax_float32_sm80(self):
        self._test_gemm_rcr_softmax(
            M=16,
            K=64,
            N=24,
            dtype="float32",
            test_name="gemm_rcr_softmax_fp32_1",
        )


filter_test_cases_by_test_env(GEMMSoftmaxTestCase)


if __name__ == "__main__":
    unittest.main()
