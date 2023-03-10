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

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)

from parameterized import parameterized


_LOGGER = logging.getLogger(__name__)


class PadGemmWithCatTestCase(unittest.TestCase):
    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_pad_gemm_rcr_with_cat(self, dtype):
        target = detect_target()
        if dtype == "float32" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M = 128
        N = 32
        K1 = 3
        K2 = 10
        K = K1 + K2

        X1 = Tensor(shape=[M, K1], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[N, K], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[N], dtype=dtype, name="b1", is_input=True)

        X2 = Tensor(shape=[M, K2], dtype=dtype, name="x2", is_input=True)
        W2 = Tensor(shape=[N, K], dtype=dtype, name="w2", is_input=True)
        B2 = Tensor(shape=[N], dtype=dtype, name="b2", is_input=True)

        X3 = ops.elementwise(FuncEnum.ADD)(X1, X1)
        X4 = ops.concatenate()([X2, X3], dim=1)
        X5 = ops.gemm_rcr_bias()(X4, W1, B1)
        X6 = ops.gemm_rcr_bias()(X4, W2, B2)
        Y = ops.concatenate()([X5, X6], dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        if int(target._arch) < 80:
            _LOGGER.warning("Skip this test on SM75")
            return
        dll_name = "test_rcr.so"
        module = compile_model(
            [Y], target, "./tmp", f"pad_gemm_with_cat_{dtype}", dll_name=dll_name
        )

        X1_pt = get_random_torch_tensor([M, K1], dtype)
        X2_pt = get_random_torch_tensor([M, K2], dtype)
        W1_pt = get_random_torch_tensor([N, K], dtype)
        W2_pt = get_random_torch_tensor([N, K], dtype)
        B1_pt = get_random_torch_tensor([N], dtype)
        B2_pt = get_random_torch_tensor([N], dtype)
        X3_pt = torch.add(X1_pt, X1_pt)
        X4_pt = torch.cat([X2_pt, X3_pt], dim=1)
        X5_pt = torch.nn.functional.linear(X4_pt, W1_pt, bias=B1_pt)
        X6_pt = torch.nn.functional.linear(X4_pt, W2_pt, bias=B2_pt)
        Y_pt = torch.cat([X5_pt, X6_pt], dim=1)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = [0] * 6
        name_to_idx = module.get_input_name_to_index_map()
        inputs[name_to_idx["x1"]] = X1_pt
        inputs[name_to_idx["x2"]] = X2_pt

        inputs[name_to_idx["w1"]] = W1_pt
        inputs[name_to_idx["w2"]] = W2_pt

        inputs[name_to_idx["b1"]] = B1_pt
        inputs[name_to_idx["b2"]] = B2_pt

        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
