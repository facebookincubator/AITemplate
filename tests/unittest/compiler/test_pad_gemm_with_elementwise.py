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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils
from parameterized import param, parameterized


class PadGemmWithElementwise(unittest.TestCase):
    @parameterized.expand(
        [
            param("static_M", [23], 7, 3),
            param("dynamic_M", [1, 78, 99], 7, 3),
        ]
    )
    def test_pad_gemm_rcr_bias_broadcast_with_elementwise(self, test_name, ms, n, k):
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        X1 = Tensor(shape=[m_dim, k], dtype="float16", name="x1", is_input=True)
        W1 = Tensor(shape=[n, k], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[n], dtype="float16", name="b1", is_input=True)
        S1 = Tensor(shape=[m_dim, n], dtype="float16", name="s1", is_input=True)
        S2 = Tensor(shape=[m_dim, n], dtype="float16", name="s2", is_input=True)

        X2 = ops.gemm_rcr_bias_mul_add()(X1, W1, B1, S1, S2)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            [Y], target, "./tmp", f"pad_gemm_with_elementwise_{test_name}"
        )

        for m in ms:
            X1_pt = torch.randn(m, k).cuda().half()
            W1_pt = torch.randn(n, k).cuda().half()
            B1_pt = torch.randn(n).cuda().half()
            S1_pt = torch.randn(m, n).cuda().half()
            S2_pt = torch.randn(m, n).cuda().half()

            X2_pt = torch.nn.functional.linear(X1_pt, W1_pt, B1_pt) * S1_pt + S2_pt
            Y_pt = X2_pt + X2_pt

            inputs = [0] * 5
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            inputs[name_to_idx["s1"]] = S1_pt
            inputs[name_to_idx["s2"]] = S2_pt
            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        [
            ("static_shape", [3], [1], 5, 3),
            ("dynamic_M", [3], [1, 78, 99], 7, 3),
            ("dynamic_B", [3, 5, 8], [3], 11, 15),
            ("dynamic_BM", [3, 5, 8], [3, 9, 10], 17, 21),
        ]
    )
    def test_pad_bmm_rrr_add_with_elementwise(self, test_name, bs, ms, n, k):
        b_dim = shape_utils.gen_int_var_min_max(bs, "B")
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        X1 = Tensor(shape=[b_dim, m_dim, k], dtype="float16", name="x1", is_input=True)
        W1 = Tensor(shape=[b_dim, k, n], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, m_dim, n], dtype="float16", name="b1", is_input=True)

        X2 = ops.bmm_rrr_add()(X1, W1, B1)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            [Y], target, "./tmp", f"pad_bmm_with_elementwise_{test_name}"
        )

        for b, m in itertools.product(bs, ms):
            X1_pt = torch.randn(b, m, k).cuda().half()
            W1_pt = torch.randn(b, k, n).cuda().half()
            B1_pt = torch.randn(b, m, n).cuda().half()

            X2_pt = torch.matmul(X1_pt, W1_pt) + B1_pt
            Y_pt = X2_pt + X2_pt

            inputs = [0, 0, 0]
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        [
            ("static_shape", [3], [1], 5, 3),
            ("dynamic_M", [3], [1, 78, 99], 7, 3),
            ("dynamic_B", [3, 5, 8], [3], 11, 15),
            ("dynamic_BM", [3, 5, 8], [3, 9, 10], 17, 21),
        ]
    )
    def test_pad_perm102_bmm_rrr_with_elementwise(self, test_name, bs, ms, n, k):
        b_dim = shape_utils.gen_int_var_min_max(bs, "B")
        m_dim = shape_utils.gen_int_var_min_max(ms, "M")

        # (M, B, K) * (B, K, N) = (M, B, N)
        X1 = Tensor(shape=[m_dim, b_dim, k], dtype="float16", name="x1", is_input=True)
        W1 = Tensor(shape=[b_dim, k, n], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[b_dim, n], dtype="float16", name="b1", is_input=True)

        X2 = ops.perm102_bmm_rrr_bias()(X1, W1, B1)
        Y = ops.elementwise(FuncEnum.ADD)(X2, X2)

        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            [Y], target, "./tmp", f"pad_perm102_with_elementwise_{test_name}"
        )

        for b, m in itertools.product(bs, ms):
            X1_pt = torch.randn(m, b, k).cuda().half()
            W1_pt = torch.randn(b, k, n).cuda().half()
            B1_pt = torch.randn(b, n).cuda().half()
            Bias_pt = B1_pt.unsqueeze(1)

            X2_pt = torch.permute(
                torch.baddbmm(Bias_pt, torch.permute(X1_pt, (1, 0, 2)), W1_pt),
                (1, 0, 2),
            )
            Y_pt = X2_pt + X2_pt
            inputs = [0, 0, 0]
            name_to_idx = module.get_input_name_to_index_map()
            inputs[name_to_idx["x1"]] = X1_pt
            inputs[name_to_idx["w1"]] = W1_pt
            inputs[name_to_idx["b1"]] = B1_pt
            y = torch.empty(Y_pt.size()).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
