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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import graph_utils, logger


class RefineGraphTestCase(unittest.TestCase):
    def test_elementwise_ops(self):
        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X2",
            is_input=True,
        )

        Y0 = (X0 + X1) * X2
        Y1 = (X0 + X2) * X1
        Y0._attrs["name"] = "Y0"
        Y0._attrs["is_output"] = True
        Y1._attrs["name"] = "Y1"
        Y1._attrs["is_output"] = True
        target = detect_target()
        module = compile_model(
            [Y0, Y1],
            target,
            "./tmp",
            "test_refine_graph_elementwise",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)

        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] != sorted_ops[1]._attrs["name"]

    def test_elementwise_ops_single_input_no_refine(self):
        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X1",
            is_input=True,
        )

        Y0 = X0 / (X0 * X0)
        Y1 = (X1 * X1) / X1
        Y0._attrs["name"] = "Y0"
        Y0._attrs["is_output"] = True
        Y1._attrs["name"] = "Y1"
        Y1._attrs["is_output"] = True
        target = detect_target()
        module = compile_model(
            [Y0, Y1],
            target,
            "./tmp",
            "test_refine_graph_elementwise",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)

        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] != sorted_ops[1]._attrs["name"]

    def test_elementwise_ops_single_input(self):
        M = 10
        N = 4
        X0 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[M, N],
            dtype="float16",
            name="X1",
            is_input=True,
        )

        Y0 = ops.elementwise(FuncEnum.SILU)(X0)
        Y1 = ops.elementwise(FuncEnum.SILU)(X1)
        Y0._attrs["name"] = "Y0"
        Y0._attrs["is_output"] = True
        Y1._attrs["name"] = "Y1"
        Y1._attrs["is_output"] = True
        target = detect_target()
        module = compile_model(
            [Y0, Y1],
            target,
            "./tmp",
            "test_elementwise_ops_single_input",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)

        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] == sorted_ops[1]._attrs["name"]

        inputs = {}
        outputs = {}
        inputs["X0"] = torch.randn([M, N]).cuda().half()
        inputs["X1"] = torch.randn([M, N]).cuda().half()
        outputs["Y0"] = torch.empty([M, N]).cuda().half()
        outputs["Y1"] = torch.empty([M, N]).cuda().half()

        module.run_with_tensors(inputs, outputs)
        y0 = torch.nn.functional.silu(inputs["X0"])
        y1 = torch.nn.functional.silu(inputs["X1"])

        self.assertTrue(torch.allclose(y0, outputs["Y0"], 1e-2, 1e-2))
        self.assertTrue(torch.allclose(y1, outputs["Y1"], 1e-2, 1e-2))

    def _build_gemm_rcr_bias(self, M, N, K, start_idx=0):
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(
            shape=X_shape, dtype="float16", name=f"input_{start_idx}", is_input=True
        )
        input_1 = Tensor(
            shape=W_shape, dtype="float16", name=f"input_{start_idx + 1}", is_input=True
        )
        input_2 = Tensor(
            shape=B_shape, dtype="float16", name=f"input_{start_idx + 2}", is_input=True
        )

        gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
        bias_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)

        return bias_tensor

    def _build_gemm_rcr_bias_mul(self, M, N, K, start_idx=0):
        D_shape = [M, N]
        input_3 = Tensor(
            shape=D_shape, dtype="float16", name=f"input_{start_idx + 3}", is_input=True
        )

        bias_tensor = self._build_gemm_rcr_bias(M, N, K, start_idx)
        mul_tensor = ops.elementwise(FuncEnum.MUL)(bias_tensor, input_3)

        return mul_tensor

    def test_gemm_ops(self):
        M = 128
        N = 64
        K = 256

        Y1 = self._build_gemm_rcr_bias_mul(M, N, K, 0)
        Y2 = self._build_gemm_rcr_bias_mul(M, N, K, 4)
        Y1._attrs["name"] = "Y0"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "Y1"
        Y2._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            [Y1, Y2],
            target,
            "./tmp",
            "test_refine_graph_gemm",
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)

        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] == sorted_ops[1]._attrs["name"]

    def test_bmm_ops_accessor(self):
        dtype = "float16"
        B = 16
        M = 128
        K = 64
        N = 256
        T_A = Tensor(
            shape=[B, M, K],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        T_B = Tensor(
            shape=[B, N, K],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        Xs = ops.split()(T_A, 32, -1)
        Ys = ops.split()(T_B, 32, -1)
        assert len(Xs) == len(Ys)

        n = len(Xs)
        Cs = []
        for i in range(n):
            X = Xs[i]
            Y = Ys[i]
            C = ops.bmm_rcr()(X, Y)
            Cs.append(C)
        Y = ops.concatenate()(Cs, dim=-1)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            "test_refine_graph_bmm",
        )

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)

        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] != sorted_ops[1]._attrs["name"]

    def test_refine_graph_group_gemms(self):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return
        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        Y1, Y2 = ops.group_gemm_rcr()(operand_groups=[[X1, W1], [X2, W2]])
        Y3, Y4 = ops.group_gemm_rcr()(operand_groups=[[X1, W1], [X2, W2]])
        Y1._attrs["name"] = "y1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "y2"
        Y2._attrs["is_output"] = True
        Y3._attrs["name"] = "y3"
        Y3._attrs["is_output"] = True
        Y4._attrs["name"] = "y4"
        Y4._attrs["is_output"] = True

        graph_outputs = [Y1, Y2, Y3, Y4]

        module = compile_model(
            graph_outputs, target, "./tmp", "test_refine_graph_group_gemms"
        )

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        assert len(sorted_ops) == 2
        assert sorted_ops[0]._attrs["name"] != sorted_ops[1]._attrs["name"]

        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
        }
        y1 = torch.empty([M, N1]).cuda().half()
        y2 = torch.empty([M, N2]).cuda().half()
        y3 = torch.empty([M, N1]).cuda().half()
        y4 = torch.empty([M, N2]).cuda().half()
        outputs = {"y1": y1, "y2": y2, "y3": y3, "y4": y4}

        module.run_with_tensors(inputs, outputs)
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y2_pt, y2, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y1_pt, outputs["y3"], atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y2_pt, outputs["y4"], atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
