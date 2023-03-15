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
)
from parameterized import param, parameterized


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GroupGEMMRcrBiasCatTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            param("group_gemm_rcr_bias_cat_fp16", "float16"),
            param("group_gemm_rcr_bias_cat_fp32_sm80", "float32"),
            param("group_gemm_rcr_bias_cat_bf16", "bfloat16"),
        ]
    )
    def test_group_gemm_rcr_bias_cat(self, test_name, dtype):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        X1 = Tensor(shape=[M, K1], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype=dtype, name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype=dtype, name="w2", is_input=True)
        B1 = Tensor(shape=[N1], dtype=dtype, name="b1", is_input=True)
        B2 = Tensor(shape=[N2], dtype=dtype, name="b2", is_input=True)
        OP = ops.group_gemm_rcr_bias()
        Y = OP(operand_groups=[[X1, W1, B1], [X2, W2, B2]], output_stride_dim=1)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        module = compile_model([Y], target, "./tmp", test_name)
        X1_pt = get_random_torch_tensor(shape=(M, K1), dtype=dtype)
        X2_pt = get_random_torch_tensor(shape=(M, K2), dtype=dtype)
        W1_pt = get_random_torch_tensor(shape=(N1, K1), dtype=dtype)
        W2_pt = get_random_torch_tensor(shape=(N2, K2), dtype=dtype)
        B1_pt = get_random_torch_tensor(shape=(N1,), dtype=dtype)
        B2_pt = get_random_torch_tensor(shape=(N2,), dtype=dtype)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt], dim=1)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        torch.testing.assert_close(y_shape, list(Y_pt.shape))

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "b1": B1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "b2": B2_pt,
        }
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(Y_pt, y, atol=1e-1, rtol=1e-1)


filter_test_cases_by_test_env(GroupGEMMRcrBiasCatTestCase)

if __name__ == "__main__":
    unittest.main()
