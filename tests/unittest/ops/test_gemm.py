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
from parameterized import parameterized


def _tolerance_limits(dtype):
    if dtype == "float16":
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype in ("float", "float32"):
        return {"atol": 3e-2, "rtol": 3e-2}
    elif dtype == "bfloat16":
        return {"atol": 2e-1, "rtol": 2e-1}
    else:
        return {}


class GEMMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(GEMMTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_rcr(self, ms, k, n, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = _tolerance_limits(dtype)
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rcr_{test_name}_{self._test_id}"
        )
        self._test_id += 1
        for m in ms:
            X_pt = get_random_torch_tensor([m, k], dtype)
            W_pt = get_random_torch_tensor([n, k], dtype)
            Y_pt = torch.nn.functional.linear(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = get_torch_empty_tensor([m, n], dtype)
            module.run_with_tensors(inputs, [y])
            if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                pass
            else:
                print(f"Processing m={m}")
                torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    def test_rcr_simple_static(self) -> None:
        self._test_rcr([1024], 256, 512, "static")

    @unittest.skipIf(detect_target().name() != "cuda", "Only supported by CUDA.")
    @parameterized.expand(
        [
            ("dynamic1", [1, 1024], 256, 512),
            # TODO/FIXME: Fix the issue below.
            # There is some bug with floating point rounding,
            # e.g. the list of batch sizes like this [1, 99, 84, 987, 1024]
            # is not handled properly.
            ("dynamic2", [1, 99, 84, 1024], 128, 8),
            ("zero_k", [8], 0, 4),
            ("zero_m", [0], 8, 4),
        ]
    )
    def test_rcr_simple_dynamic(self, name, ms, k, n) -> None:
        self._test_rcr(ms, k, n, name)

    def _test_rcr_dynamic_n(self, ms, k, ns, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = _tolerance_limits(dtype)
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ns), k],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rcr_{test_name}_{self._test_id}"
        )
        self._test_id += 1

        for m in ms:
            for n in ns:
                X_pt = get_random_torch_tensor([m, k], dtype)
                W_pt = get_random_torch_tensor([n, k], dtype)
                Y_pt = torch.nn.functional.linear(X_pt, W_pt)

                inputs = {"input_0": X_pt, "input_1": W_pt}
                y = get_torch_empty_tensor([m, n], dtype)
                module.run_with_tensors(inputs, [y])

                if X_pt.nelement() == 0 or W_pt.nelement() == 0:
                    pass
                else:
                    torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    def test_rcr_dynamic_n(self):
        self._test_rcr([16, 1 * 29, 64], 256, 300000, "einsum_1")
        self._test_rcr_dynamic_n(
            [16, 1 * 29, 64], 256, [100000, 300000], "einsum_dynamic_n"
        )

    def _test_3d_2d_rcr(self, m0s, m1s, k, n, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = _tolerance_limits(dtype)
        if dtype == "float16":
            tolerance_limits["atol"] = 2e-2
            tolerance_limits["rtol"] = 2e-2
        X = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(m0s),
                shape_utils.gen_int_var_min_max(m1s),
                k,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        X._attrs["is_input"] = True
        W = Tensor(shape=[n, k], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_3d_2d_rcr_{test_name}_{self._test_id}"
        )
        self._test_id += 1

        for m0, m1 in itertools.product(m0s, m1s):
            X_pt = get_random_torch_tensor([m0, m1, k], dtype)
            W_pt = get_random_torch_tensor([n, k], dtype)
            Y_pt = torch.nn.functional.linear(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = get_torch_empty_tensor([m0, m1, n], dtype)
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_3d_2d_rcr(self):
        self._test_3d_2d_rcr([1024], [2], 256, 512, "static")
        self._test_3d_2d_rcr([1, 1024], [2], 256, 512, "dynamic1")
        self._test_3d_2d_rcr([3], [128, 256], 256, 512, "dynamic2")
        self._test_3d_2d_rcr([1, 99, 1024], [1, 2], 128, 8, "dynamic3")

    def _test_rrr(self, ms, k, n, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = _tolerance_limits(dtype)
        if dtype == "float16":
            tolerance_limits["atol"] = 2e-2
            tolerance_limits["rtol"] = 2e-2
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rrr_{test_name}_{self._test_id}"
        )
        self._test_id += 1

        for m in ms:
            X_pt = get_random_torch_tensor([m, k], dtype)
            W_pt = get_random_torch_tensor([k, n], dtype)
            Y_pt = torch.matmul(X_pt, W_pt)
            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = get_torch_empty_tensor([m, n], dtype)
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    def test_rrr(self):
        self._test_rrr([256], 128, 32, "static")
        if detect_target().name() == "cuda":
            self._test_rrr([1, 99, 1024, 2048], 256, 16, "dynamic")

    def _test_3d_2d_rrr(self, m0s, m1s, k, n, test_name, dtype="float16"):
        target = detect_target()
        tolerance_limits = {"atol": 2e-1, "rtol": 2e-1}
        X = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(m0s),
                shape_utils.gen_int_var_min_max(m1s),
                k,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"gemm_rrr_{test_name}_{self._test_id}"
        )
        self._test_id += 1

        for m0, m1 in itertools.product(m0s, m1s):
            X_pt = get_random_torch_tensor([m0, m1, k], dtype)
            W_pt = get_random_torch_tensor([k, n], dtype)
            Y_pt = torch.matmul(X_pt, W_pt)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = get_torch_empty_tensor([m0, m1, n], dtype)
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(Y_pt, y, **tolerance_limits)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_3d_2d_rrr(self):
        self._test_3d_2d_rrr([256], [2], 128, 32, "static")
        self._test_3d_2d_rrr([1, 128], [3], 256, 16, "dynamic1")
        self._test_3d_2d_rrr([2], [24, 36], 256, 16, "dynamic2")
        self._test_3d_2d_rrr([2, 34, 48], [1, 3, 5], 256, 16, "dynamic3")

    @parameterized.expand(("float16", "float32", "bfloat16"))
    def test_h_rcr(self, ait_dtype):
        M = 256
        K = 256
        N = 512
        target = detect_target(use_fp16_acc=(ait_dtype == "float16"))
        if target.name() != "cuda" and ait_dtype != "float16":
            self.skipTest(
                f"{ait_dtype} input type is not supported for {target.name()}"
            )
        if (
            target.name() == "cuda"
            and int(target._arch) < 80
            and ait_dtype != "float16"
        ):
            self.skipTest(f"{ait_dtype} is not supported for cuda sm < 80")
        X = Tensor(shape=[M, K], dtype=ait_dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=ait_dtype, name="input_1", is_input=True)
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(
            Y, target, "./tmp", f"hgemm_rcr_{ait_dtype}_{self._test_id}"
        )
        self._test_id += 1
        X_pt = get_random_torch_tensor((M, K), ait_dtype)
        W_pt = get_random_torch_tensor((N, K), ait_dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt}
        y = get_torch_empty_tensor((M, N), ait_dtype)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(Y_pt, y, atol=1e-1, rtol=1e-1)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_gemm_float(self):
        self._test_rcr([1024], 256, 512, "static_float", dtype="float")
        self._test_rcr([1, 1024], 256, 512, "dynamic1_float", dtype="float")
        self._test_rcr([16, 1 * 29, 64], 256, 300000, "einsum_1_float", dtype="float")

        self._test_3d_2d_rcr([1024], [2], 256, 512, "static_float", dtype="float")
        self._test_3d_2d_rcr(
            [1, 99, 1024], [1, 2], 128, 8, "dynamic3_float", dtype="float"
        )

        self._test_rrr([256], 128, 32, "static_float", dtype="float")
        self._test_rrr([1, 99, 1024, 2048], 256, 16, "dynamic_float", dtype="float")

        self._test_3d_2d_rrr([256], [2], 128, 32, "static_float", dtype="float")
        self._test_3d_2d_rrr(
            [2, 34, 48], [1, 3, 5], 256, 16, "dynamic3_float", dtype="float"
        )

    @unittest.skipIf(
        detect_target().name() == "rocm", "bfloat16 is not supported by ROCm."
    )
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "bfloat16 is not supported by CUDA < SM80.",
    )
    def test_gemm_bfloat16(self):
        self._test_rcr([1024], 256, 512, "static_bfloat16", dtype="bfloat16")
        self._test_rcr([1, 1024], 256, 512, "dynamic1_bfloat16", dtype="bfloat16")
        self._test_rcr(
            [16, 1 * 29, 64], 256, 300000, "einsum_1_bfloat16", dtype="bfloat16"
        )

        self._test_3d_2d_rcr([1024], [2], 256, 512, "static_bfloat16", dtype="bfloat16")
        self._test_3d_2d_rcr(
            [1, 99, 1024], [1, 2], 128, 8, "dynamic3_bfloat16", dtype="bfloat16"
        )

        self._test_rrr([256], 128, 32, "static_bfloat16", dtype="bfloat16")
        self._test_rrr(
            [1, 99, 1024, 2048], 256, 16, "dynamic_bfloat16", dtype="bfloat16"
        )

        self._test_3d_2d_rrr([256], [2], 128, 32, "static_bfloat16", dtype="bfloat16")
        self._test_3d_2d_rrr(
            [2, 34, 48], [1, 3, 5], 256, 16, "dynamic3_bfloat16", dtype="bfloat16"
        )


if __name__ == "__main__":
    unittest.main()
