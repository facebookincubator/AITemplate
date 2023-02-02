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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMPermuteTestCase(unittest.TestCase):
    def _test_rrr(self, bs, ms, N, K, d1, test_name, copy_op=False, dtype="float16"):
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X = Tensor(
            shape=[batch_dim, m_dim, K], dtype=dtype, name="input_0", is_input=True
        )
        W = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="input_1", is_input=True)
        OP = ops.bmm_rrr_permute(shape=(d1,))
        if copy_op:
            OP = ops.bmm_rrr_permute(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_rrr_{}".format(test_name))

        for (b, m) in itertools.product(bs, ms):
            X_pt = get_random_torch_tensor([b, m, K], dtype)
            W_pt = get_random_torch_tensor([b, K, N], dtype)

            Y_l = torch.bmm(X_pt, W_pt)
            Y_r = Y_l.reshape(b // d1, d1, m, N)
            Y_pt = torch.permute(Y_r, [0, 2, 1, 3])

            y = get_torch_empty_tensor(Y_pt.shape, dtype)
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_rrr(self):
        self._test_rrr([24], [80], N=88, K=64, d1=12, test_name="permute1")
        self._test_rrr([10240], [88], N=88, K=64, d1=10, test_name="permute2")
        self._test_rrr([100], [88], N=88, K=64, d1=10, test_name="permute3")
        self._test_rrr(
            [100], [88], N=88, K=64, d1=10, test_name="permute3_copy_op", copy_op=True
        )
        if detect_target().name() != "rocm":
            self._test_rrr([24], [80], N=0, K=96, d1=12, test_name="permute1_zero_n")
            self._test_rrr([24], [0], N=32, K=96, d1=12, test_name="permute1_zero_m")

    def _test_rcr(self, bs, ms, N, K, d1, test_name, copy_op=False, dtype="float16"):
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X = Tensor(
            shape=[batch_dim, m_dim, K], dtype=dtype, name="input_0", is_input=True
        )
        W = Tensor(shape=[batch_dim, N, K], dtype=dtype, name="input_1", is_input=True)
        OP = ops.bmm_rcr_permute(shape=(d1,))
        if copy_op:
            OP = ops.bmm_rcr_permute(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_rcr_{}".format(test_name))

        for (b, m) in itertools.product(bs, ms):
            X_pt = get_random_torch_tensor([b, m, K], dtype)
            W_pt = get_random_torch_tensor([b, N, K], dtype)

            WT = torch.transpose(W_pt, 2, 1)
            Y_l = torch.bmm(X_pt, WT)
            Y_r = Y_l.reshape(b // d1, d1, m, N)
            Y_pt = torch.permute(Y_r, [0, 2, 1, 3])

            y = get_torch_empty_tensor(Y_pt.shape, dtype)
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_rcr(self):
        self._test_rcr([10240], [88], N=64, K=88, d1=10, test_name="permute1")
        self._test_rcr([24], [80], N=64, K=88, d1=12, test_name="permute2")
        self._test_rcr([100], [88], N=64, K=88, d1=10, test_name="permute3")
        self._test_rcr(
            [100], [88], N=64, K=88, d1=10, test_name="permute3_copy_op", copy_op=True
        )
        if detect_target().name() != "rocm":
            self._test_rcr(
                [0], [80], N=96, K=32, d1=12, test_name="permute1_zero_batch"
            )
            self._test_rcr([24], [80], N=96, K=0, d1=12, test_name="permute1_zero_k")

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_permute_float(self):
        self._test_rrr(
            [10], [8], N=88, K=64, d1=10, test_name="permute3_float", dtype="float"
        )
        self._test_rrr(
            [10],
            [8],
            N=88,
            K=64,
            d1=10,
            test_name="permute3_copy_op_float",
            copy_op=True,
            dtype="float",
        )
        self._test_rcr(
            [10], [8], N=64, K=88, d1=10, test_name="permute3_float", dtype="float"
        )

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_permute_bfloat16(self):
        self._test_rrr(
            [10],
            [8],
            N=88,
            K=64,
            d1=10,
            test_name="permute3_bfloat16",
            dtype="bfloat16",
        )
        self._test_rrr(
            [10],
            [8],
            N=88,
            K=64,
            d1=10,
            test_name="permute3_copy_op_bfloat16",
            copy_op=True,
            dtype="bfloat16",
        )
        self._test_rcr(
            [10],
            [8],
            N=64,
            K=88,
            d1=10,
            test_name="permute3_bfloat16",
            dtype="bfloat16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
