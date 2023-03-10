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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


_LOGGER = logging.getLogger(__name__)


class PadBmmBiasWithCatTestCase(unittest.TestCase):
    def _test_pad_bmm_rrr_bias_with_cat(
        self, test_name, bs, ms, n, k1, k2, dtype="float16"
    ):
        k = k1 + k2
        b_dim = shape_utils.gen_int_var_min_max(bs, name="b")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X1 = Tensor(shape=[b_dim, m_dim, k1], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[b_dim, m_dim, k2], dtype=dtype, name="x2", is_input=True)
        X4 = ops.concatenate()([X1, X2], dim=2)

        W1 = Tensor(shape=[b_dim, k, n], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, m_dim, n], dtype=dtype, name="b1", is_input=True)
        W2 = Tensor(shape=[b_dim, k, n], dtype=dtype, name="w2", is_input=True)
        B2 = Tensor(shape=[b_dim, m_dim, n], dtype=dtype, name="b2", is_input=True)
        Y1 = ops.bmm_rrr_add()(X4, W1, B1)
        Y2 = ops.bmm_rrr_add()(X4, W2, B2)

        Y = ops.concatenate()([Y1, Y2], dim=2)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Skip this test on SM75")
            return
        module = compile_model(
            [Y], target, "./tmp", f"test_bmm_rrr_padding_{test_name}_{dtype}"
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))

        for b, m in itertools.product(bs, ms):
            X1_pt = get_random_torch_tensor([b, m, k1], dtype)
            X2_pt = get_random_torch_tensor([b, m, k2], dtype)
            X4_pt = torch.cat([X1_pt, X2_pt], dim=2)

            W1_pt = get_random_torch_tensor([b, k, n], dtype)
            B1_pt = get_random_torch_tensor([b, m, n], dtype)
            W2_pt = get_random_torch_tensor([b, k, n], dtype)
            B2_pt = get_random_torch_tensor([b, m, n], dtype)

            Y1_pt = torch.baddbmm(B1_pt, X4_pt, W1_pt)
            Y2_pt = torch.baddbmm(B2_pt, X4_pt, W2_pt)
            Y_pt = torch.cat([Y1_pt, Y2_pt], dim=2)

            inputs = [0] * 6
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["x2"]] = X2_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["w2"]] = W2_pt
            inputs[name_to_idx["b1"]] = B1_pt
            inputs[name_to_idx["b2"]] = B2_pt

            y = get_torch_empty_tensor(Y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_pad_bmm_rrr_bias_with_cat_float16(self):
        self._test_pad_bmm_rrr_bias_with_cat(
            "static_odd_k", bs=[2], ms=[64], n=32, k1=3, k2=10
        )
        self._test_pad_bmm_rrr_bias_with_cat(
            "static_odd_kn", bs=[2], ms=[128], n=31, k1=1, k2=8
        )
        self._test_pad_bmm_rrr_bias_with_cat(
            "dynamic_odd_kn", bs=[1, 2, 3], ms=[2, 5, 7], n=15, k1=1, k2=2
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_pad_bmm_rrr_bias_with_cat_float32_sm80(self):
        self._test_pad_bmm_rrr_bias_with_cat(
            "static_odd_k", bs=[2], ms=[64], n=32, k1=3, k2=10, dtype="float32"
        )
        self._test_pad_bmm_rrr_bias_with_cat(
            "dynamic_odd_kn",
            bs=[1, 2, 3],
            ms=[2, 5, 7],
            n=15,
            k1=1,
            k2=2,
            dtype="float32",
        )


filter_test_cases_by_test_env(PadBmmBiasWithCatTestCase)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
