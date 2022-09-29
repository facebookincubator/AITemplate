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
from aitemplate.utils import shape_utils


class GEMMTestCase(unittest.TestCase):
    def _test_rcr(self, ms, k, n, test_name):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rcr_{}".format(test_name))

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(n, k).cuda().half()
            Y_pt = torch.nn.functional.linear(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty([m, n]).cuda().half()
            module.run_with_tensors(inputs, [y])
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        self._test_rcr([1024], 256, 512, "static")
        if detect_target().name() == "cuda":
            self._test_rcr([1, 1024], 256, 512, "dynamic1")
            self._test_rcr([1, 99, 84, 987, 1024], 128, 8, "dynamic2")
            self._test_rcr([8], 0, 4, "zero_k")
            self._test_rcr([0], 8, 4, "zero_m")

    def _test_3d_2d_rcr(self, m0s, m1s, k, n, test_name):
        target = detect_target()
        X = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(m0s),
                shape_utils.gen_int_var_min_max(m1s),
                k,
            ],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        X._attrs["is_input"] = True
        W = Tensor(shape=[n, k], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", "gemm_3d_2d_rcr_{}".format(test_name)
        )

        for m0, m1 in itertools.product(m0s, m1s):
            X_pt = torch.randn(m0, m1, k).cuda().half()
            W_pt = torch.randn(n, k).cuda().half()
            Y_pt = torch.nn.functional.linear(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty([m0, m1, n]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_3d_2d_rcr(self):
        self._test_3d_2d_rcr([1024], [2], 256, 512, "static")
        self._test_3d_2d_rcr([1, 1024], [2], 256, 512, "dynamic1")
        self._test_3d_2d_rcr([3], [128, 256], 256, 512, "dynamic2")
        self._test_3d_2d_rcr([1, 99, 1024], [1, 2], 128, 8, "dynamic3")

    def _test_rrr(self, ms, k, n, test_name):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rrr_{}".format(test_name))

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(k, n).cuda().half()
            Y_pt = torch.matmul(X_pt, W_pt)
            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty([m, n]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr([256], 128, 32, "static")
        if detect_target().name() == "cuda":
            self._test_rrr([1, 99, 1024, 2048], 256, 16, "dynamic")

    def _test_3d_2d_rrr(self, m0s, m1s, k, n, test_name):
        target = detect_target()
        X = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(m0s),
                shape_utils.gen_int_var_min_max(m1s),
                k,
            ],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rrr_{}".format(test_name))

        for m0, m1 in itertools.product(m0s, m1s):
            X_pt = torch.randn(m0, m1, k).cuda().half()
            W_pt = torch.randn(k, n).cuda().half()
            Y_pt = torch.matmul(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty([m0, m1, n]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_3d_2d_rrr(self):
        self._test_3d_2d_rrr([256], [2], 128, 32, "static")
        self._test_3d_2d_rrr([1, 128], [3], 256, 16, "dynamic1")
        self._test_3d_2d_rrr([2], [24, 36], 256, 16, "dynamic2")
        self._test_3d_2d_rrr([2, 34, 48], [1, 3, 5], 256, 16, "dynamic3")

    def test_h_rcr(self):
        M = 256
        K = 256
        N = 512
        target = detect_target(use_fp16_acc=True)
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "hgemm_rcr")
        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        Y_pt = torch.nn.functional.linear(X_pt, W_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt}
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
