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
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


_LOGGER = logging.getLogger(__name__)


class PadGemmWithCatTestCase(unittest.TestCase):
    def _test_pad_gemm_rrr_with_cat(self, test_name, ms, n, k1, k2, dtype="float16"):
        k = k1 + k2
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X1 = Tensor(shape=[m_dim, k1], dtype=dtype, name="x1", is_input=True)
        W1 = Tensor(shape=[k, n], dtype=dtype, name="w1", is_input=True)
        X2 = Tensor(shape=[m_dim, k2], dtype=dtype, name="x2", is_input=True)
        W2 = Tensor(shape=[k, n], dtype=dtype, name="w2", is_input=True)
        X4 = ops.concatenate()([X1, X2], dim=1)
        Y1 = ops.gemm_rrr()(X4, W1)
        Y2 = ops.gemm_rrr()(X4, W2)
        Y = ops.concatenate()([Y1, Y2], dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Skip this test on SM75")
            return
        dll_name = f"test_rrr_padding_{test_name}.so"
        module = compile_model(
            [Y], target, "./tmp", f"pad_gemm_with_cat_rrr_{dtype}", dll_name=dll_name
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))

        for m in ms:
            X1_pt = get_random_torch_tensor([m, k1], dtype)
            W1_pt = get_random_torch_tensor([k, n], dtype)
            X2_pt = get_random_torch_tensor([m, k2], dtype)
            W2_pt = get_random_torch_tensor([k, n], dtype)
            X4_pt = torch.cat([X1_pt, X2_pt], dim=1)
            Y1_pt = torch.matmul(X4_pt, W1_pt)
            Y2_pt = torch.matmul(X4_pt, W2_pt)
            Y_pt = torch.cat([Y1_pt, Y2_pt], dim=1)

            inputs = [0] * 4
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["x2"]] = X2_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["w2"]] = W2_pt
            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_pad_gemm_rrr_with_cat_float16(self):
        self._test_pad_gemm_rrr_with_cat("static_odd_k", ms=[128], n=32, k1=3, k2=10)
        self._test_pad_gemm_rrr_with_cat("static_odd_kn", ms=[128], n=31, k1=1, k2=8)
        self._test_pad_gemm_rrr_with_cat(
            "dynamic_odd_kn", ms=[2, 5, 7], n=15, k1=1, k2=2
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_pad_gemm_rrr_with_cat_float32_sm80(self):
        self._test_pad_gemm_rrr_with_cat(
            "static_odd_k", ms=[128], n=32, k1=3, k2=10, dtype="float32"
        )
        self._test_pad_gemm_rrr_with_cat(
            "dynamic_odd_kn",
            ms=[2, 5, 7],
            n=15,
            k1=1,
            k2=2,
            dtype="float32",
        )


filter_test_cases_by_test_env(PadGemmWithCatTestCase)

if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
