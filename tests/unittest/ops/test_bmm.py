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


class BMMTestCase(unittest.TestCase):
    def _test_rcr(self, bs, ms, N, K, test_name):
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X = Tensor(
            shape=[batch_dim, m_dim, K], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[batch_dim, N, K], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.bmm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_rcr_{}".format(test_name))

        for (b, m) in itertools.product(bs, ms):
            X_pt = torch.randn(b, m, K).cuda().half()
            W_pt = torch.randn(b, N, K).cuda().half()

            WT = torch.transpose(W_pt, 2, 1)
            Y_pt = torch.bmm(X_pt, WT)

            y = torch.empty([b, m, N]).cuda().half()
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            if X_pt.nelement() == 0 or Y_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_rcr(self):
        self._test_rcr([1024], [128], N=512, K=256, test_name="static")
        if detect_target().name() == "cuda":
            self._test_rcr([1, 5, 977, 1024], [32], N=512, K=256, test_name="dynamic_b")
            self._test_rcr([1], [100, 200, 300], N=512, K=256, test_name="dynamic_m")
            self._test_rcr(
                [1, 2, 5], [100, 200, 300], N=512, K=256, test_name="dynamic_bm"
            )
            self._test_rcr([0], [128], N=512, K=256, test_name="zero_batch")
            self._test_rcr([1], [128], N=512, K=0, test_name="zero_k")
            self._test_rcr([1], [128], N=0, K=8, test_name="zero_n")

    def _test_crr(self, bs, ks, test_name):
        M = 256
        N = 512
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        k_dim = shape_utils.gen_int_var_min_max(ks, name="k")
        X = Tensor(
            shape=[batch_dim, k_dim, M], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[batch_dim, k_dim, N], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.bmm_crr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_crr_{}".format(test_name))

        for (b, k) in itertools.product(bs, ks):
            X_pt = torch.randn(b, k, M).cuda().half()
            W_pt = torch.randn(b, k, N).cuda().half()

            XT = torch.transpose(X_pt, 2, 1)
            Y_pt = torch.bmm(XT, W_pt)

            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_crr(self):
        self._test_crr([1024], [128], "static")
        if detect_target().name() == "cuda":
            self._test_crr([3, 977, 1024], [128], "dynamic_b")
            self._test_crr([5], [45, 56, 78], "dynamic_k")
            self._test_crr([1, 2, 5], [3, 6, 8], "dynamic_bk")

    def _test_rrr(self, bs, ms, test_name):
        K = 256
        N = 512
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X = Tensor(
            shape=[batch_dim, m_dim, K], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[batch_dim, K, N], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.bmm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_rrr_{}".format(test_name))

        for (b, m) in itertools.product(bs, ms):
            X_pt = torch.randn(b, m, K).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()

            Y_pt = torch.bmm(X_pt, W_pt)

            y = torch.empty([b, m, N]).cuda().half()
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_rrr(self):
        self._test_rrr([87], [23], "static")
        if detect_target().name() == "cuda":
            self._test_rrr([2, 5, 99], [23], "dynamic_b")
            self._test_rrr([77], [4, 7, 9], "dynamic_m")
            self._test_rrr([2, 5, 7], [1, 7, 9], "dynamic_bm")

    def _test_ccr(self, bs, test_name):
        M = 256
        N = 64
        K = 128
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        X = Tensor(
            shape=[batch_dim, K, M], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[batch_dim, N, K], dtype="float16", name="input_1", is_input=True
        )
        OP = ops.bmm_ccr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "bmm_ccr_{}".format(test_name))

        for b in bs:
            X_pt = torch.randn(b, K, M).cuda().half()
            W_pt = torch.randn(b, N, K).cuda().half()

            XT = torch.transpose(X_pt, 2, 1)
            Y_pt = torch.bmm(XT, W_pt.transpose(2, 1))
            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_ccr(self):
        self._test_ccr([77], "static")
        if detect_target().name() == "cuda":
            self._test_ccr([1, 9, 101], "dynamic_b")


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMBroadcastTestCase(unittest.TestCase):
    def test_rcr_with_accessors(self):
        A_shape = [2, 2, 4]
        B_shape = [2, 8, 4]
        C_shape = [2, 2, 8]

        X_expanded = Tensor(
            shape=A_shape, dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        C = Tensor(shape=C_shape, dtype="float16", name="input_2", is_input=True)

        X, _ = ops.split()(X_expanded, [1, 1], 0)
        OP = ops.bmm_rcr()
        Y = OP(X, W)
        out = ops.concatenate()([Y, C], 0)
        out._attrs["name"] = "output_0"
        out._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(out, target, "./tmp", "bmm_rcr_with_accessor")

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()
        C_pt = torch.randn(*C_shape).cuda().half()

        X_feed, _ = torch.split(X_pt, [1, 1], 0)

        WT = torch.transpose(W_pt, -2, -1)
        Y_pt = torch.matmul(X_feed, WT)
        out_pt = torch.cat((Y_pt, C_pt), dim=0)

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0, 0, 0]
        inputs[input_name_to_index["input_0"]] = X_pt
        inputs[input_name_to_index["input_1"]] = W_pt
        inputs[input_name_to_index["input_2"]] = C_pt
        y = torch.empty([4, 2, 8]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(out_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr_merge_with_accessors(self):
        A_shape = [2, 2, 4]
        B_shape = [4, 8, 4]

        X_expanded = Tensor(
            shape=A_shape, dtype="float16", name="input_0", is_input=True
        )
        W_expanded = Tensor(
            shape=B_shape, dtype="float16", name="input_1", is_input=True
        )

        X1, X2 = ops.split()(X_expanded, [1, 1], 0)
        W1, W2 = ops.split()(W_expanded, [2, 2], 0)
        Y1 = ops.bmm_rcr()(X1, W1)
        Y2 = ops.bmm_rcr()(X2, W2)
        out = ops.concatenate()([Y1, Y2], 0)
        out._attrs["name"] = "output_0"
        out._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(out, target, "./tmp", "bmm_rcr_merge_with_accessor")

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()

        X1_pt, X2_pt = torch.split(X_pt, [1, 1], 0)

        WT = torch.transpose(W_pt, -2, -1)
        W1_pt, W2_pt = torch.split(WT, [2, 2], 0)
        Y1_pt = torch.matmul(X1_pt, W1_pt)
        Y2_pt = torch.matmul(X2_pt, W2_pt)
        out_pt = torch.cat((Y1_pt, Y2_pt), dim=0)

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0, 0]
        inputs[input_name_to_index["input_0"]] = X_pt
        inputs[input_name_to_index["input_1"]] = W_pt
        y = torch.empty([4, 2, 8]).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(out_pt, y, atol=1e-1, rtol=1e-1))

    def _test_rcr(self, A_shape, B_shape, test_name):
        M, N = A_shape[-2], B_shape[-2]
        if len(A_shape) == 2:
            B = B_shape[0]
        elif len(B_shape) == 2:
            B = A_shape[0]
        else:
            B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_rcr_{}".format(test_name))

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()

        WT = torch.transpose(W_pt, -2, -1)
        Y_pt = torch.matmul(X_pt, WT)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        self._test_rcr([1, 16, 8], [2, 32, 8], "broadcastable_a")
        self._test_rcr([2, 16, 8], [1, 32, 8], "broadcastable_b")
        self._test_rcr([16, 8], [8, 32, 8], "2d_broadcastable_a")
        self._test_rcr([8, 16, 8], [32, 8], "2d_broadcastable_b")

    def _test_crr(self, A_shape, B_shape, test_name):
        M, N = A_shape[-1], B_shape[-1]
        if len(A_shape) == 2:
            B = B_shape[0]
        elif len(B_shape) == 2:
            B = A_shape[0]
        else:
            B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_crr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_crr_{}".format(test_name))

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()

        XT = torch.transpose(X_pt, -2, -1)
        Y_pt = torch.matmul(XT, W_pt)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_crr(self):
        self._test_crr([1, 8, 16], [2, 8, 32], "broadcastable_a")
        self._test_crr([2, 8, 16], [1, 8, 32], "broadcastable_b")
        self._test_crr([8, 16], [8, 8, 32], "2d_broadcastable_a")
        self._test_crr([8, 8, 16], [8, 32], "2d_broadcastable_b")

    def _test_rrr(self, A_shape, B_shape, test_name):
        M, N = A_shape[-2], B_shape[-1]
        if len(A_shape) == 2:
            B = B_shape[0]
        elif len(B_shape) == 2:
            B = A_shape[0]
        else:
            B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_rrr_{}".format(test_name))

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()

        Y_pt = torch.matmul(X_pt, W_pt)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr([1, 16, 8], [2, 8, 32], "broadcastable_a")
        self._test_rrr([2, 16, 8], [1, 8, 32], "broadcastable_b")
        self._test_rrr([16, 8], [8, 8, 32], "2d_broadcastable_a")
        self._test_rrr([8, 16, 8], [8, 32], "2d_broadcastable_b")

    def _test_ccr(self, A_shape, B_shape, test_name):
        M, N = A_shape[-1], B_shape[-2]
        if len(A_shape) == 2:
            B = B_shape[0]
        elif len(B_shape) == 2:
            B = A_shape[0]
        else:
            B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_ccr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_ccr_{}".format(test_name))

        X_pt = torch.randn(*A_shape).cuda().half()
        W_pt = torch.randn(*B_shape).cuda().half()

        XT = torch.transpose(X_pt, -2, -1)
        WT = torch.transpose(W_pt, -2, -1)
        Y_pt = torch.matmul(XT, WT)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_ccr(self):
        self._test_ccr([1, 8, 16], [2, 32, 8], "broadcastable_a")
        self._test_ccr([2, 8, 16], [1, 32, 8], "broadcastable_b")
        self._test_ccr([8, 16], [8, 32, 8], "2d_broadcastable_a")
        self._test_ccr([8, 8, 16], [32, 8], "2d_broadcastable_b")


if __name__ == "__main__":
    unittest.main()
