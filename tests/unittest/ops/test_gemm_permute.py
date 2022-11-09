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
from aitemplate.utils import shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMTestCase(unittest.TestCase):
    def _test_rcr(self, ms, k, n, shape, test_name, has_bias=False, copy_op=False):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[n], dtype="float16", name="input_2", is_input=True)
        if has_bias:
            OP = ops.gemm_rcr_bias_permute(shape)
            if copy_op:
                OP = ops.gemm_rcr_bias_permute(**OP._get_op_attributes())
            Y = OP(X, W, B)
        else:
            OP = ops.gemm_rcr_permute(shape)
            if copy_op:
                OP = ops.gemm_rcr_permute(**OP._get_op_attributes())
            Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"gemm_rcr_{test_name}")

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(n, k).cuda().half()
            B_pt = torch.randn(n).cuda().half()
            if has_bias:
                Y_l = torch.nn.functional.linear(X_pt, W_pt, B_pt)
            else:
                Y_l = torch.nn.functional.linear(X_pt, W_pt)
            Y_r = Y_l.reshape(16, *shape, 16)
            Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])

            inputs = {"input_0": X_pt, "input_1": W_pt}
            if has_bias:
                inputs["input_2"] = B_pt
            y = torch.empty(Y_pt.shape).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr(self):
        for has_bias in (True, False):
            for copy_op in (True, False):
                self._test_rcr(
                    [80],
                    32,
                    96,
                    (5, 3, 2),
                    "permute1",
                    has_bias=has_bias,
                    copy_op=copy_op,
                )
                self._test_rcr(
                    [128],
                    64,
                    256,
                    (8, 4, 4),
                    "permute2",
                    has_bias=has_bias,
                    copy_op=copy_op,
                )

    def _test_rcr_0213(
        self, ms, k, n, shape, test_name, has_bias=False, copy_op=False, layout="0213"
    ):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[n], dtype="float16", name="input_2", is_input=True)
        if has_bias:
            OP = ops.gemm_rcr_bias_permute(shape, layout)
            if copy_op:
                OP = ops.gemm_rcr_bias_permute(**OP._get_op_attributes())
            Y = OP(X, W, B)
        else:
            OP = ops.gemm_rcr_permute(shape, layout)
            if copy_op:
                OP = ops.gemm_rcr_permute(**OP._get_op_attributes())
            Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"gemm_rcr_{test_name}")

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(n, k).cuda().half()
            B_pt = torch.randn(n).cuda().half()

            def torch_f(x, w, b, has_bias, shape):
                if has_bias:
                    Y_l = torch.nn.functional.linear(x, w, b)
                else:
                    Y_l = torch.nn.functional.linear(x, w)
                t1, t2 = shape
                Y_r = Y_l.reshape(m // t1, t1, t2, n // t2)
                Y_pt = torch.permute(Y_r, [0, 2, 1, 3])
                Y_out = Y_pt.reshape([m // t1, t2, -1])
                return Y_pt, Y_out

            Y_pt, _ = torch_f(X_pt, W_pt, B_pt, has_bias, shape)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            if has_bias:
                inputs["input_2"] = B_pt
            y = torch.empty(Y_pt.shape).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

            # module.benchmark_with_tensors(inputs, [y], count=1000)
            # from aitemplate.testing.benchmark_pt import benchmark_torch_function

            # t = benchmark_torch_function(
            #     1000, torch_f, X_pt, W_pt, B_pt, has_bias, shape
            # )
            # print(f"pt: {t} ms/iter")

    def test_rcr_0213(self):
        self._test_rcr_0213(
            [54],
            256,
            4000000,
            [54, 1000000],
            "permute_0213_1",
            has_bias=False,
            copy_op=False,
            layout="0213",
        )
        self._test_rcr_0213(
            [29, 29 * 8],
            256,
            300000,
            [29, 100000],
            "permute_0213_2",
            has_bias=False,
            copy_op=False,
            layout="0213",
        )

    def _test_rrr(self, ms, k, n, shape, test_name, copy_op=False):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[k, n], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr_permute(shape)
        if copy_op:
            OP = ops.gemm_rrr_permute(**OP._get_op_attributes())
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "gemm_rrr_{}".format(test_name))

        for m in ms:
            X_pt = torch.randn(m, k).cuda().half()
            W_pt = torch.randn(k, n).cuda().half()
            Y_l = torch.matmul(X_pt, W_pt)
            Y_r = Y_l.reshape(16, *shape, 16)
            Y_pt = torch.permute(Y_r, [2, 0, 3, 1, 4])
            inputs = {"input_0": X_pt, "input_1": W_pt}
            y = torch.empty(Y_pt.shape).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_rrr(self):
        self._test_rrr([80], 32, 96, (5, 3, 2), "permute1")
        self._test_rrr([128], 64, 256, (8, 4, 4), "permute2")
        self._test_rrr([128], 64, 256, (8, 4, 4), "permute2_copy_op", copy_op=True)


if __name__ == "__main__":
    unittest.main()
