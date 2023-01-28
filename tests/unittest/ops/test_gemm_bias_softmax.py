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
import os
import unittest

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


_LOGGER = logging.getLogger(__name__)


# @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
@unittest.skip("GEMM + Softmax is disabled for now")
class GEMMTestCase(unittest.TestCase):
    def _test_gemm_rcr_bias_softmax(
        self, M=16, K=64, N=24, rebuild=True, dtype="float16"
    ):
        target = detect_target()
        if type(target).__name__ == "FBCUDA":
            _LOGGER.warning("Skip this test for special profiling requirement")
            return

        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_softmax()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = torch.softmax(Y_pt, dim=1)
        Y_np = Y_pt.cpu().numpy()

        test_name = f"gemm_bias_softmax_{dtype}"
        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))
        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        np.testing.assert_allclose(
            np.argmax(Y_np, axis=1),
            np.argmax(y.cpu().numpy(), axis=1),
            atol=1e-1,
            rtol=1e-1,
        )

    def test_gemm_bias_softmax(self):
        self._test_gemm_rcr_bias_softmax(N=81)

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_gemm_bias_softmax_float(self):
        self._test_gemm_rcr_bias_softmax(N=81, dtype="float")


if __name__ == "__main__":
    unittest.main()
