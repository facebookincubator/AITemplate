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
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class BMMAddTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(BMMAddTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_rrr(self, B, M, K, N, dtype="float16"):
        target = detect_target()
        X = Tensor(shape=[B, M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, K, N], dtype=dtype, name="input_1", is_input=True)
        D = Tensor(shape=[B, M, N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.bmm_rrr_add()
        Y = OP(X, W, D)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", f"bmm_rrr_add_{dtype}", dll_name=dll_name
        )
        X_pt = get_random_torch_tensor([B, M, K], dtype)
        W_pt = get_random_torch_tensor([B, K, N], dtype)
        D_pt = get_random_torch_tensor([B, M, N], dtype)

        Y_pt = torch.bmm(X_pt, W_pt)
        Y_pt = Y_pt + D_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": D_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.test_count += 1

    def _test_ccr(self, B, M, N, K, test_name, dtype="float16"):
        target = detect_target()
        X = Tensor(shape=[B, K, M], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="input_1", is_input=True)
        D = Tensor(shape=[B, M, N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.bmm_ccr_add()
        Y = OP(X, W, D)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        X_pt = get_random_torch_tensor([B, K, M], dtype)
        W_pt = get_random_torch_tensor([B, N, K], dtype)
        D_pt = get_random_torch_tensor([B, M, N], dtype)

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt.transpose(2, 1))
        Y_pt = Y_pt + D_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": D_pt}, [y]
        )
        if X_pt.nelement() == 0 or W_pt.nelement == 0:
            pass
        else:
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def _test_crr(self, B, M, K, N, dtype="float16"):
        target = detect_target()
        X = Tensor(
            shape=[B, K, M],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[B, K, N],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        D = Tensor(
            shape=[B, M, N],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.bmm_crr_add()
        Y = OP(X, W, D)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        test_name = f"bmm_crr_add_{dtype}"
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        X_pt = get_random_torch_tensor([B, K, M], dtype)
        W_pt = get_random_torch_tensor([B, K, N], dtype)
        D_pt = get_random_torch_tensor([B, M, N], dtype)

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt)
        Y_pt = Y_pt + D_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": D_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_rrr(self):
        self._test_rrr(B=32, M=256, K=256, N=512)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_ccr(self):
        self._test_ccr(B=32, M=256, N=256, K=512, test_name="bmm_ccr_add")
        self._test_ccr(B=0, M=256, N=256, K=512, test_name="bmm_ccr_zero_batch")
        self._test_ccr(B=1, M=0, N=256, K=512, test_name="bmm_ccr_zero_m")
        self._test_ccr(B=1, M=256, N=256, K=0, test_name="bmm_ccr_zero_k")

    def test_crr(self):
        self._test_crr(B=32, M=256, K=256, N=512)

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_add_float(self):
        self._test_rrr(B=8, M=32, K=8, N=64, dtype="float")
        self._test_ccr(
            B=8, M=32, N=64, K=16, test_name="bmm_ccr_add_float", dtype="float"
        )
        self._test_crr(B=8, M=32, K=16, N=64, dtype="float")

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_add_bfloat16(self):
        self._test_rrr(B=8, M=32, K=8, N=64, dtype="bfloat16")
        self._test_ccr(
            B=8, M=32, N=64, K=16, test_name="bmm_ccr_add_bfloat16", dtype="bfloat16"
        )
        self._test_crr(B=8, M=32, K=16, N=64, dtype="bfloat16")


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMBroadcastTestCase(unittest.TestCase):
    def _test_crr(self, A_shape, B_shape, bias_shape, test_name, dtype="float16"):
        M, N = A_shape[-1], B_shape[-1]
        B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        bias = Tensor(shape=bias_shape, dtype=dtype, name="input_2", is_input=True)
        Y = ops.bmm_crr_add()(X, W, bias)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_crr_{}".format(test_name))

        X_pt = get_random_torch_tensor(A_shape, dtype)
        W_pt = get_random_torch_tensor(B_shape, dtype)
        bias_pt = get_random_torch_tensor(bias_shape, dtype)

        XT = torch.transpose(X_pt, -2, -1)
        Y_pt = torch.matmul(XT, W_pt) + bias_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": bias_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_crr(self):
        self._test_crr(
            [1, 8, 16], [2, 8, 32], bias_shape=[32], test_name="broadcastable_bias1d"
        )
        self._test_crr(
            [1, 8, 16],
            [2, 8, 32],
            bias_shape=[1, 32],
            test_name="broadcastable_bias1d_2",
        )
        self._test_crr(
            [1, 8, 16],
            [2, 8, 32],
            bias_shape=[16, 32],
            test_name="broadcastable_bias2d",
        )
        self._test_crr(
            [1, 8, 16],
            [2, 8, 32],
            bias_shape=[1, 16, 32],
            test_name="broadcastable_bias3d",
        )

    def _test_rrr(self, A_shape, B_shape, bias_shape, test_name, dtype="float16"):
        M, N = A_shape[-2], B_shape[-1]
        B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        bias = Tensor(shape=bias_shape, dtype=dtype, name="input_2", is_input=True)
        Y = ops.bmm_rrr_add()(X, W, bias)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_rrr_{}".format(test_name))

        X_pt = get_random_torch_tensor(A_shape, dtype)
        W_pt = get_random_torch_tensor(B_shape, dtype)
        bias_pt = get_random_torch_tensor(bias_shape, dtype)

        Y_pt = torch.matmul(X_pt, W_pt) + bias_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": bias_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr(
            [1, 16, 8], [2, 8, 32], bias_shape=[32], test_name="broadcastable_bias1d"
        )
        self._test_rrr(
            [1, 16, 8],
            [2, 8, 32],
            bias_shape=[1, 32],
            test_name="broadcastable_bias1d_2",
        )
        self._test_rrr(
            [1, 16, 8],
            [2, 8, 32],
            bias_shape=[16, 32],
            test_name="broadcastable_bias2d",
        )
        self._test_rrr(
            [1, 16, 8],
            [2, 8, 32],
            bias_shape=[1, 16, 32],
            test_name="broadcastable_bias3d",
        )

    def _test_ccr(self, A_shape, B_shape, bias_shape, test_name, dtype="float16"):
        M, N = A_shape[-1], B_shape[-2]
        B = max(A_shape[0], B_shape[0])

        X = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        bias = Tensor(shape=bias_shape, dtype=dtype, name="input_2", is_input=True)
        Y = ops.bmm_ccr_add()(X, W, bias)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y, target, "./tmp", "bmm_ccr_{}".format(test_name))

        X_pt = get_random_torch_tensor(A_shape, dtype)
        W_pt = get_random_torch_tensor(B_shape, dtype)
        bias_pt = get_random_torch_tensor(bias_shape, dtype)

        XT = torch.transpose(X_pt, -2, -1)
        WT = torch.transpose(W_pt, -2, -1)
        Y_pt = torch.matmul(XT, WT) + bias_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": bias_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_ccr(self):
        self._test_ccr(
            [1, 8, 16], [2, 32, 8], bias_shape=[32], test_name="broadcastable_bias1d"
        )
        self._test_ccr(
            [1, 8, 16],
            [2, 32, 8],
            bias_shape=[1, 32],
            test_name="broadcastable_bias1d_2",
        )
        self._test_ccr(
            [1, 8, 16],
            [2, 32, 8],
            bias_shape=[16, 32],
            test_name="broadcastable_bias2d",
        )
        self._test_ccr(
            [1, 8, 16],
            [2, 32, 8],
            bias_shape=[1, 16, 32],
            test_name="broadcastable_bias3d",
        )

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_add_broadcast_float(self):
        self._test_crr(
            [1, 8, 16],
            [2, 8, 32],
            bias_shape=[16, 32],
            test_name="broadcastable_bias2d_float",
            dtype="float",
        )
        self._test_rrr(
            [1, 16, 8],
            [2, 8, 32],
            bias_shape=[1, 32],
            test_name="broadcastable_bias1d_2_float",
            dtype="float",
        )
        self._test_ccr(
            [1, 8, 16],
            [2, 32, 8],
            bias_shape=[1, 16, 32],
            test_name="broadcastable_bias3d_float",
            dtype="float",
        )

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_bmm_add_broadcast_bfloat16(self):
        self._test_crr(
            [1, 8, 16],
            [2, 8, 32],
            bias_shape=[16, 32],
            test_name="broadcastable_bias2d_bfloat16",
            dtype="bfloat16",
        )
        self._test_rrr(
            [1, 16, 8],
            [2, 8, 32],
            bias_shape=[1, 32],
            test_name="broadcastable_bias1d_2_bfloat16",
            dtype="bfloat16",
        )
        self._test_ccr(
            [1, 8, 16],
            [2, 32, 8],
            bias_shape=[1, 16, 32],
            test_name="broadcastable_bias3d_bfloat16",
            dtype="bfloat16",
        )


if __name__ == "__main__":
    unittest.main()
