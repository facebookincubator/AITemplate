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
class BMMSoftmaxTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_bmm_rcr_softmax(
        self,
        B=16,
        M=16,
        K=64,
        N=24,
        dtype="float16",
        test_name="bmm_rcr_softmax",
    ):
        X = Tensor(shape=[B, M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="input_1", is_input=True)
        OP = ops.bmm_rcr_softmax()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        x_pt = get_random_torch_tensor([B, M, K], dtype)
        w_pt = get_random_torch_tensor([B, N, K], dtype)
        wt_pt = torch.transpose(w_pt, 2, 1)
        y_pt = torch.bmm(x_pt, wt_pt)
        y_pt = torch.softmax(y_pt, dim=-1)

        module = compile_model(Y, detect_target(), "./tmp", test_name)

        inputs = {"input_0": x_pt, "input_1": w_pt}
        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(inputs, [y])

        torch.testing.assert_close(y, y_pt, atol=1e-2, rtol=1e-2)

        torch.testing.assert_close(
            torch.argmax(y, axis=2),
            torch.argmax(y_pt, axis=2),
            atol=1e-1,
            rtol=1e-1,
        )

    def test_bmm_rcr_softmax_float16(self):
        self._test_bmm_rcr_softmax(
            B=16,
            M=16,
            K=64,
            N=24,
            dtype="float16",
            test_name="bmm_rcr_softmax_fp16_1",
        )

    def test_bmm_rcr_softmax_float32_sm80(self):
        self._test_bmm_rcr_softmax(
            B=16,
            M=16,
            K=64,
            N=24,
            dtype="float32",
            test_name="bmm_rcr_softmax_fp32_1",
        )


filter_test_cases_by_test_env(BMMSoftmaxTestCase)


if __name__ == "__main__":
    unittest.main()
