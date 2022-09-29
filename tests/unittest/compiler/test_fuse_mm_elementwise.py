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
from aitemplate.utils import shape_utils


class FuseGemmRcrBiasCase(unittest.TestCase):
    def _build_gemm_rcr_bias(self, M, N, K, decomposed):
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype="float16", name="input_2", is_input=True)

        if decomposed:
            gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
            bias_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        else:
            bias_tensor = ops.gemm_universal.gemm_rcr_bias()(input_0, input_1, input_2)

        return bias_tensor

    def _build_gemm_rcr_bias_add_add_relu_chain(self, M, N, K, depth, decomposed):
        D_shape = [M, N]
        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)
        input_4 = Tensor(shape=D_shape, dtype="float16", name="input_4", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed)
        if depth == 1:
            return bias_tensor

        add_tensor = ops.elementwise(FuncEnum.ADD)(bias_tensor, input_3)
        if depth == 2:
            return add_tensor

        add2_tensor = ops.elementwise(FuncEnum.ADD)(add_tensor, input_4)
        if depth == 3:
            return add2_tensor

        relu_tensor = ops.elementwise(FuncEnum.RELU)(add2_tensor)
        if depth == 4:
            return relu_tensor

        raise AssertionError("No suitable output tensors available")

    def _build_gemm_rcr_bias_mul(self, M, N, K, decomposed):
        D_shape = [M, N]
        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed)
        mul_tensor = ops.elementwise(FuncEnum.MUL)(bias_tensor, input_3)

        return mul_tensor

    def _test_gemm_rcr_bias(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        bias_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 1, decomposed
        )
        bias_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(bias_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            y = torch.empty([M, N]).cuda().half()

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        add_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 2, decomposed
        )
        add_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_add")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D0_pt)

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt
            y = torch.empty([M, N]).cuda().half()

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_add(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        add2_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 3, decomposed
        )
        add2_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(add2_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_add_add")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            D1_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D0_pt + D1_pt
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt
            inputs[input_name_to_index["input_4"]] = D1_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_add_relu(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        relu_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 4, decomposed
        )
        relu_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(relu_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_add_add_relu")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            D1_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.nn.functional.relu(
                    torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D0_pt + D1_pt
                )
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt
            inputs[input_name_to_index["input_4"]] = D1_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_add_fail(self):
        M, N, K = 16, 32, 8
        B_shape = [N]

        input_3 = Tensor(shape=B_shape, dtype="float16", name="input_3", is_input=True)

        gemm_bias_tensor = self._build_gemm_rcr_bias(M, N, K, False)
        gemm_bias_tensor._attrs["name"] = "gemm_tensor"
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_bias_tensor, input_3)
        add_tensor._attrs["name"] = "gemm_bias_add_tensor"

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", "gemm_bias_fusion_add_fail")

        # This shouldn't be merged into gemm_rcr_bias_add since input_3 needs broadcasting
        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "gemm_tensor":
                check_tensor = tensor
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias")

        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N).cuda().half()
        B1_pt = torch.randn(N).cuda().half()
        Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + B1_pt)

        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors([X_pt, W_pt, B_pt, B1_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_chained(self):
        M, N, K = 16, 32, 8
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype="float16", name="input_2", is_input=True)

        gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        add_tensor._attrs["name"] = "first_gemm"

        D_shape = [N, N]
        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)
        gemm1_tensor = ops.gemm_universal.gemm_rcr()(add_tensor, input_3)
        add1_tensor = ops.elementwise(FuncEnum.ADD)(gemm1_tensor, input_2)
        add1_tensor._attrs["name"] = "second_gemm"

        output = ops.elementwise(FuncEnum.COS)(add1_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", "gemm_bias_fusion_chained")

        gemm_check = [False, False]
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "first_gemm":
                src_op = list(tensor.src_ops())[0]
                self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias")
                gemm_check[0] = True
            if tensor._attrs["name"] == "second_gemm":
                src_op = list(tensor.src_ops())[0]
                self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias")
                gemm_check[1] = True
        self.assertTupleEqual(tuple(gemm_check), (True, True))

        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N).cuda().half()
        D_pt = torch.randn(N, N).cuda().half()
        Y_pt = torch.cos(
            torch.nn.functional.linear(
                torch.nn.functional.linear(X_pt, W_pt, bias=B_pt), D_pt, bias=B_pt
            )
        )

        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors([X_pt, W_pt, B_pt, D_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_fail(self):
        M, N, K = 16, 32, 8
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [M, N]

        input_0 = Tensor(shape=X_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype="float16", name="input_2", is_input=True)

        gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        add_tensor._attrs["name"] = "final_tensor"

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", "gemm_bias_fusion_fail")

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if len(tensor.src_ops()) != 1:
                continue
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"] == "gemm_rcr":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)

        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(M, N).cuda().half()
        Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt) + B_pt)

        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors([X_pt, W_pt, B_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_relu(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]

        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed)
        add_tensor = ops.elementwise(FuncEnum.ADD)(bias_tensor, input_3)
        relu_tensor = ops.elementwise(FuncEnum.RELU)(add_tensor)
        relu_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(relu_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_add_relu")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.nn.functional.relu(
                    torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D0_pt
                )
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 4
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_tanh(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed)
        tanh_tensor = ops.elementwise(FuncEnum.TANH)(bias_tensor)
        tanh_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(tanh_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_tanh")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.cos(
                torch.tanh(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed)
        mul_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(mul_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_mul")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) * D0_pt)

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 4
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul_add(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]

        input_4 = Tensor(shape=D_shape, dtype="float16", name="input_4", is_input=True)
        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed)
        add_tensor = ops.elementwise(FuncEnum.ADD)(mul_tensor, input_4)
        add_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_mul_add")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            D1_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) * D0_pt + D1_pt
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 5
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt
            inputs[input_name_to_index["input_4"]] = D1_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul_tanh(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed)
        tanh_tensor = ops.elementwise(FuncEnum.TANH)(mul_tensor)
        tanh_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(tanh_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_mul_tanh")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D0_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.tanh(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) * D0_pt)
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 4
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias(self):
        self._test_gemm_rcr_bias([8], 16, 8, True, "gemm_rcr_bias_basic_decomposed")
        self._test_gemm_rcr_bias([8], 16, 8, False, "gemm_rcr_bias_basic")
        self._test_gemm_rcr_bias([8, 32], 16, 8, False, "gemm_rcr_bias_dynamic")
        self._test_gemm_rcr_bias([8], 16, 3, False, "gemm_rcr_bias_need_align")

    def test_gemm_rcr_bias_add(self):
        self._test_gemm_rcr_bias_add(
            [8], 16, 8, True, "gemm_rcr_bias_add_basic_decomposed"
        )
        self._test_gemm_rcr_bias_add([8], 16, 8, False, "gemm_rcr_bias_add_basic")
        self._test_gemm_rcr_bias_add([8, 32], 16, 8, False, "gemm_rcr_bias_add_dynamic")
        self._test_gemm_rcr_bias_add([8], 16, 3, False, "gemm_rcr_bias_add_need_align")

    def test_gemm_rcr_bias_add_add(self):
        self._test_gemm_rcr_bias_add_add(
            [8], 16, 8, True, "gemm_rcr_bias_add_add_basic_decomposed"
        )
        self._test_gemm_rcr_bias_add_add(
            [8], 16, 8, False, "gemm_rcr_bias_add_add_basic"
        )
        self._test_gemm_rcr_bias_add_add(
            [8, 32], 16, 8, False, "gemm_rcr_bias_add_add_dynamic"
        )
        self._test_gemm_rcr_bias_add_add(
            [8], 16, 3, False, "gemm_rcr_bias_add_add_need_align"
        )

    def test_gemm_rcr_bias_add_add_relu(self):
        self._test_gemm_rcr_bias_add_add_relu(
            [8], 16, 8, True, "gemm_rcr_bias_add_add_relu_basic_decomposed"
        )
        self._test_gemm_rcr_bias_add_add_relu(
            [8], 16, 8, False, "gemm_rcr_bias_add_add_relu_basic"
        )
        self._test_gemm_rcr_bias_add_add_relu(
            [8, 32], 16, 8, False, "gemm_rcr_bias_add_add_relu_dynamic"
        )
        self._test_gemm_rcr_bias_add_add_relu(
            [8], 16, 3, False, "gemm_rcr_bias_add_add_relu_need_align"
        )

    def test_gemm_rcr_bias_add_relu(self):
        self._test_gemm_rcr_bias_add_relu(
            [8], 16, 8, True, "gemm_rcr_bias_add_relu_basic_decomposed"
        )
        self._test_gemm_rcr_bias_add_relu(
            [8], 16, 8, False, "gemm_rcr_bias_add_relu_basic"
        )
        self._test_gemm_rcr_bias_add_relu(
            [8, 32], 16, 8, False, "gemm_rcr_bias_add_relu_dynamic"
        )
        self._test_gemm_rcr_bias_add_relu(
            [8], 16, 3, False, "gemm_rcr_bias_add_relu_need_align"
        )

    def test_gemm_rcr_bias_mul(self):
        self._test_gemm_rcr_bias_mul(
            [8], 16, 8, True, "gemm_rcr_bias_mul_basic_decomposed"
        )
        self._test_gemm_rcr_bias_mul([8], 16, 8, False, "gemm_rcr_bias_mul_basic")
        self._test_gemm_rcr_bias_mul([8, 32], 16, 8, False, "gemm_rcr_bias_mul_dynamic")
        self._test_gemm_rcr_bias_mul([8], 16, 3, False, "gemm_rcr_bias_mul_need_align")

    def test_gemm_rcr_bias_mul_add(self):
        self._test_gemm_rcr_bias_mul_add(
            [8], 16, 8, True, "gemm_rcr_bias_mul_add_basic_decomposed"
        )
        self._test_gemm_rcr_bias_mul_add(
            [8], 16, 8, False, "gemm_rcr_bias_mul_add_basic"
        )
        self._test_gemm_rcr_bias_mul_add(
            [8, 32], 16, 8, False, "gemm_rcr_bias_mul_add_dynamic"
        )
        self._test_gemm_rcr_bias_mul_add(
            [8], 16, 3, False, "gemm_rcr_bias_mul_add_need_align"
        )

    def test_gemm_rcr_bias_mul_tanh(self):
        self._test_gemm_rcr_bias_mul_tanh(
            [8], 16, 8, True, "gemm_rcr_bias_mul_tanh_basic_decomposed"
        )
        self._test_gemm_rcr_bias_mul_tanh(
            [8], 16, 8, False, "gemm_rcr_bias_mul_tanh_basic"
        )
        self._test_gemm_rcr_bias_mul_tanh(
            [8, 32], 16, 8, False, "gemm_rcr_bias_mul_tanh_dynamic"
        )
        self._test_gemm_rcr_bias_mul_tanh(
            [8], 16, 3, False, "gemm_rcr_bias_mul_tanh_need_align"
        )


class FuseGemmRcrBiasActivationCase(unittest.TestCase):
    def _build_gemm_rcr_bias(self, M, N, K, decomposed):
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype="float16", name="input_2", is_input=True)

        if decomposed:
            gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
            bias_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        else:
            bias_tensor = ops.gemm_rcr_bias()(input_0, input_1, input_2)

        return bias_tensor

    def _build_gemm_rcr_bias_sigmoid(self, M, N, K, decomposed):
        gemm_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed)
        sigmoid_tensor = ops.elementwise(FuncEnum.SIGMOID)(gemm_tensor)

        return sigmoid_tensor

    def _test_gemm_rcr_bias_activation(
        self, Ms, N, K, activation, target_ait, decomposed, testname
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        if activation == "relu":
            ait_func = FuncEnum.RELU
            pt_func = torch.nn.functional.relu
        elif activation == "sigmoid":
            ait_func = FuncEnum.SIGMOID
            pt_func = torch.sigmoid
        elif activation == "tanh":
            ait_func = FuncEnum.TANH
            pt_func = torch.tanh
        else:
            raise AssertionError("Activation not supported")

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed)
        act_tensor = ops.elementwise(ait_func)(bias_tensor)
        act_tensor._attrs["name"] = "final_tensor"
        output = ops.elementwise(FuncEnum.COS)(act_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], target_ait)

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            Y_pt = torch.cos(pt_func(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)))

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            y = torch.empty([M, N]).cuda().half()

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_sigmoid_mul(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]
        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)

        sigmoid_tensor = self._build_gemm_rcr_bias_sigmoid(m_dim, N, K, decomposed)
        mul_tensor = ops.elementwise(FuncEnum.MUL)(sigmoid_tensor, input_3)
        mul_tensor._attrs["name"] = "final_tensor"

        output = ops.elementwise(FuncEnum.COS)(mul_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_sigmoid_mul")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.sigmoid(torch.nn.functional.linear(X_pt, W_pt, B_pt)) * D_pt
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_sigmoid_mul_tanh(self, Ms, N, K, decomposed, testname):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]
        input_3 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)

        sigmoid_tensor = self._build_gemm_rcr_bias_sigmoid(m_dim, N, K, decomposed)
        mul_tensor = ops.elementwise(FuncEnum.MUL)(sigmoid_tensor, input_3)
        tanh_tensor = ops.elementwise(FuncEnum.TANH)(mul_tensor)
        tanh_tensor._attrs["name"] = "final_tensor"

        output = ops.elementwise(FuncEnum.COS)(tanh_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_sigmoid_mul_tanh")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D_pt = torch.randn(M, N).cuda().half()
            Y_pt = torch.cos(
                torch.tanh(
                    torch.sigmoid(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))
                    * D_pt
                )
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_gemm_rcr_bias_relu(self):
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            8,
            "relu",
            "gemm_rcr_bias_relu",
            True,
            "gemm_rcr_bias_relu_basic_decomposed",
        )
        self._test_gemm_rcr_bias_activation(
            [8], 16, 8, "relu", "gemm_rcr_bias_relu", False, "gemm_rcr_bias_relu_basic"
        )
        self._test_gemm_rcr_bias_activation(
            [8, 32],
            16,
            8,
            "relu",
            "gemm_rcr_bias_relu",
            False,
            "gemm_rcr_bias_relu_dynamic",
        )
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            3,
            "relu",
            "gemm_rcr_bias_relu",
            False,
            "gemm_rcr_bias_relu_need_align",
        )

    def test_gemm_rcr_bias_sigmoid(self):
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            8,
            "sigmoid",
            "gemm_rcr_bias_sigmoid",
            True,
            "gemm_rcr_bias_sigmoid_basic_decomposed",
        )
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            8,
            "sigmoid",
            "gemm_rcr_bias_sigmoid",
            False,
            "gemm_rcr_bias_sigmoid_basic",
        )
        self._test_gemm_rcr_bias_activation(
            [8, 32],
            16,
            8,
            "sigmoid",
            "gemm_rcr_bias_sigmoid",
            False,
            "gemm_rcr_bias_sigmoid_dynamic",
        )
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            3,
            "sigmoid",
            "gemm_rcr_bias_sigmoid",
            False,
            "gemm_rcr_bias_sigmoid_need_align",
        )

    def test_gemm_rcr_bias_sigmoid_mul(self):
        self._test_gemm_rcr_bias_sigmoid_mul(
            [8], 16, 8, True, "gemm_rcr_bias_sigmoid_mul_basic_decomposed"
        )
        self._test_gemm_rcr_bias_sigmoid_mul(
            [8], 16, 8, False, "gemm_rcr_bias_sigmoid_mul_basic"
        )
        self._test_gemm_rcr_bias_sigmoid_mul(
            [8, 32], 16, 8, False, "gemm_rcr_bias_sigmoid_mul_dynamic"
        )
        self._test_gemm_rcr_bias_sigmoid_mul(
            [8], 16, 3, False, "gemm_rcr_bias_sigmoid_mul_need_align"
        )

    def test_gemm_rcr_bias_sigmoid_mul_tanh(self):
        self._test_gemm_rcr_bias_sigmoid_mul_tanh(
            [8], 16, 8, True, "gemm_rcr_bias_sigmoid_mul_tanh_basic_decomposed"
        )
        self._test_gemm_rcr_bias_sigmoid_mul_tanh(
            [8], 16, 8, False, "gemm_rcr_bias_sigmoid_mul_tanh_basic"
        )
        self._test_gemm_rcr_bias_sigmoid_mul_tanh(
            [8, 32], 16, 8, False, "gemm_rcr_bias_sigmoid_mul_tanh_dynamic"
        )
        self._test_gemm_rcr_bias_sigmoid_mul_tanh(
            [8], 16, 3, False, "gemm_rcr_bias_sigmoid_mul_tanh_need_align"
        )

    def test_gemm_rcr_bias_tanh(self):
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            8,
            "tanh",
            "gemm_rcr_bias_tanh",
            True,
            "gemm_rcr_bias_tanh_basic_decomposed",
        )
        self._test_gemm_rcr_bias_activation(
            [8], 16, 8, "tanh", "gemm_rcr_bias_tanh", False, "gemm_rcr_bias_tanh_basic"
        )
        self._test_gemm_rcr_bias_activation(
            [8, 32],
            16,
            8,
            "tanh",
            "gemm_rcr_bias_tanh",
            False,
            "gemm_rcr_bias_tanh_dynamic",
        )
        self._test_gemm_rcr_bias_activation(
            [8],
            16,
            3,
            "tanh",
            "gemm_rcr_bias_tanh",
            False,
            "gemm_rcr_bias_tanh_need_align",
        )


class FuseGemmRcrBiasSwishCase(unittest.TestCase):
    def _test_gemm_rcr_bias_swish(self, Ms, N, K, testname, use_add=False):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        X_shape = [m_dim, K]
        W_shape = [N, K]
        B_shape = [N]
        D_shape = [m_dim, N]
        input_1 = Tensor(shape=X_shape, dtype="float16", name="input_0", is_input=True)
        input_2 = Tensor(shape=W_shape, dtype="float16", name="input_1", is_input=True)
        input_3 = Tensor(shape=B_shape, dtype="float16", name="input_2", is_input=True)
        input_4 = Tensor(shape=D_shape, dtype="float16", name="input_3", is_input=True)

        if use_add:
            tensor = ops.gemm_rcr()(input_1, input_2)
            gemm_tensor = ops.elementwise(FuncEnum.ADD)(tensor, input_3)
        else:
            gemm_tensor = ops.gemm_rcr_bias()(input_1, input_2, input_3)
        sigmoid_tensor = ops.elementwise(FuncEnum.SIGMOID)(gemm_tensor)
        swish_tensor = ops.elementwise(FuncEnum.MUL)(gemm_tensor, sigmoid_tensor)
        swish_tensor._attrs["name"] = "final_tensor"

        output = ops.elementwise(FuncEnum.ADD)(swish_tensor, input_4)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "final_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias_swish")

        for M in Ms:
            X_pt = torch.randn(M, K).cuda().half()
            W_pt = torch.randn(N, K).cuda().half()
            B_pt = torch.randn(N).cuda().half()
            D_pt = torch.randn(M, N).cuda().half()
            gemm_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
            Y_pt = gemm_pt * torch.sigmoid(gemm_pt) + D_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D_pt

            y = torch.empty([M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_gemm_rcr_bias_swish(self):
        self._test_gemm_rcr_bias_swish([8], 16, 8, "gemm_rcr_bias_swish_basic")
        self._test_gemm_rcr_bias_swish([8, 32], 16, 8, "gemm_rcr_bias_swish_dynamic")
        self._test_gemm_rcr_bias_swish([8], 16, 3, "gemm_rcr_bias_swish_need_align")

    def test_gemm_rcr_add_swish(self):
        self._test_gemm_rcr_bias_swish([8], 16, 8, "gemm_rcr_add_swish_basic", True)
        self._test_gemm_rcr_bias_swish(
            [8, 32], 16, 8, "gemm_rcr_add_swish_dynamic", True
        )
        self._test_gemm_rcr_bias_swish(
            [8], 16, 3, "gemm_rcr_add_swish_need_align", True
        )


class FuseBmmCcrAddCase(unittest.TestCase):
    def _test_bmm_ccr_add(self, Bs, M, N, K, testname):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, K, M]
        B_shape = [batch_dim, N, K]
        D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype="float16", name="input_2", is_input=True)
        bmm_tensor = ops.gemm_universal.bmm_ccr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, input_2)
        add_tensor._attrs["name"] = "add_tensor"
        output = ops.elementwise(FuncEnum.ADD)(add_tensor, input_2)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            src_ops = list(tensor.src_ops())
            if len(src_ops) != 1:
                continue
            if src_ops[0]._attrs["op"].startswith("bmm"):
                check_tensor = tensor
                self.assertEqual(src_ops[0]._attrs["op"], "bmm_ccr_add")
                break
        self.assertIsNotNone(check_tensor)

        for B in Bs:
            X_pt = torch.randn(B, K, M).cuda().half()
            W_pt = torch.randn(B, N, K).cuda().half()
            D0_pt = torch.randn(B, M, N).cuda().half()
            Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt.transpose(2, 1)) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = torch.empty([B, M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_bmm_ccr_add_negative(self, testname, negative_type):
        B, K, M, N = 8, 32, 16, 8
        A_shape = [B, K, M]
        B_shape = [B, N, K]
        D0_shape = [B, M, N]
        input_0 = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype="float16", name="input_2", is_input=True)
        bmm_tensor = ops.gemm_universal.bmm_ccr()(input_0, input_1)
        bmm_tensor._attrs["name"] = "bmm_tensor"
        if negative_type == "is_output":
            bmm_tensor._attrs["is_output"] = True
            output_1 = bmm_tensor
        elif negative_type == "other_input":
            other_tensor = ops.elementwise(FuncEnum.COS)(bmm_tensor)
            other_tensor._attrs["name"] = "output_1"
            other_tensor._attrs["is_output"] = True
            output_1 = other_tensor
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, input_2)
        add_tensor._attrs["name"] = "add_tensor"
        output = ops.elementwise(FuncEnum.ADD)(add_tensor, input_2)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model([output, output_1], target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_tensor":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "bmm_ccr")

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        D0_pt = torch.randn(B, M, N).cuda().half()

        bmm_pt = torch.bmm(X_pt.transpose(2, 1), W_pt.transpose(2, 1))
        Y_pt = bmm_pt + D0_pt + D0_pt
        if negative_type == "is_output":
            Y1_pt = bmm_pt
        else:
            Y1_pt = torch.cos(bmm_pt)

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0, 0, 0]
        inputs[input_name_to_index["input_0"]] = X_pt
        inputs[input_name_to_index["input_1"]] = W_pt
        inputs[input_name_to_index["input_2"]] = D0_pt

        y = torch.empty([B, M, N]).cuda().half()
        y1 = torch.empty([B, M, N]).cuda().half()
        output_name_to_index = module.get_output_name_to_index_map()
        if output_name_to_index["output_0"] == 0:
            ys = [y, y1]
        else:
            ys = [y1, y]

        module.run_with_tensors(inputs, ys)
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))

    def test_bmm_ccr_add(self):
        self._test_bmm_ccr_add([8], 32, 16, 8, "bmm_ccr_add_basic")
        self._test_bmm_ccr_add([8, 32], 32, 16, 8, "bmm_ccr_add_dynamic")
        self._test_bmm_ccr_add([8], 7, 13, 3, "bmm_ccr_add_need_align")

    def test_bmm_ccr_add_negative(self):
        self._test_bmm_ccr_add_negative("bmm_ccr_add_negative_output", "is_output")
        self._test_bmm_ccr_add_negative("bmm_ccr_add_negative_input", "other_input")

    def test_bmm_ccr_add_double_shared_input(self):
        B, M, N, K = 8, 32, 16, 8

        A_shape = [B, K, M]
        B_shape = [B, N, K]
        D0_shape = [B, M, N]
        input_0 = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        input_11 = Tensor(
            shape=B_shape, dtype="float16", name="input_11", is_input=True
        )
        bmm_tensor = ops.gemm_universal.bmm_ccr()(input_0, input_1)
        bmm_tensor_1 = ops.gemm_universal.bmm_ccr()(input_0, input_11)

        input_2 = Tensor(shape=D0_shape, dtype="float16", name="input_2", is_input=True)
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, input_2)
        add_tensor._attrs["name"] = "add_tensor"
        add_tensor_1 = ops.elementwise(FuncEnum.ADD)(bmm_tensor_1, input_2)
        add_tensor_1._attrs["name"] = "add_tensor_1"

        output = ops.elementwise(FuncEnum.ADD)(add_tensor, input_2)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True
        output_1 = ops.elementwise(FuncEnum.ADD)(add_tensor_1, input_2)
        output_1._attrs["name"] = "output_1"
        output_1._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(
            [output, output_1], target, "./tmp", "bmm_ccr_double_shared_inputs"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if (
                tensor._attrs["name"] == "add_tensor"
                or tensor._attrs["name"] == "add_tensor_1"
            ):
                check_tensor = tensor
            if check_tensor is None:
                continue
            self.assertEqual(len(check_tensor.src_ops()), 1)
            src_op = list(check_tensor.src_ops())[0]
            self.assertEqual(src_op._attrs["op"], "bmm_ccr_add")
            check_tensor = None

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        W1_pt = torch.randn(B, N, K).cuda().half()
        D0_pt = torch.randn(B, M, N).cuda().half()
        Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt.transpose(2, 1)) + D0_pt + D0_pt
        Y1_pt = torch.bmm(X_pt.transpose(2, 1), W1_pt.transpose(2, 1)) + D0_pt + D0_pt

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [None] * 4
        inputs[input_name_to_index["input_0"]] = X_pt
        inputs[input_name_to_index["input_1"]] = W_pt
        inputs[input_name_to_index["input_11"]] = W1_pt
        inputs[input_name_to_index["input_2"]] = D0_pt

        y = torch.empty([B, M, N]).cuda().half()
        y1 = torch.empty([B, M, N]).cuda().half()
        ys = [None] * 2
        output_name_to_index = module.get_output_name_to_index_map()
        ys[output_name_to_index["output_0"]] = y
        ys[output_name_to_index["output_1"]] = y1

        module.run_with_tensors(inputs, ys)

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))


class FuseBmmCrrAddCase(unittest.TestCase):
    def _test_bmm_crr_add(self, Bs, M, N, K, testname):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, K, M]
        B_shape = [batch_dim, K, N]
        D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype="float16", name="input_2", is_input=True)
        bmm_tensor = ops.gemm_universal.bmm_crr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, input_2)
        add_tensor._attrs["name"] = "add_tensor"
        output = ops.elementwise(FuncEnum.ADD)(add_tensor, input_2)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            src_ops = list(tensor.src_ops())
            if len(src_ops) != 1:
                continue
            if src_ops[0]._attrs["op"].startswith("bmm"):
                check_tensor = tensor
                self.assertEqual(src_ops[0]._attrs["op"], "bmm_crr_add")
                break
        self.assertIsNotNone(check_tensor)

        for B in Bs:
            X_pt = torch.randn(B, K, M).cuda().half()
            W_pt = torch.randn(B, K, N).cuda().half()
            D0_pt = torch.randn(B, M, N).cuda().half()
            Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = torch.empty([B, M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_crr_add(self):
        self._test_bmm_crr_add([8], 32, 16, 8, "bmm_crr_add_basic")
        self._test_bmm_crr_add([8, 32], 32, 16, 8, "bmm_crr_add_dynamic")
        self._test_bmm_crr_add([8], 7, 13, 3, "bmm_crr_add_need_align")


class FuseBmmRrrAddCase(unittest.TestCase):
    def _test_bmm_rrr_add(self, Bs, M, N, K, testname):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, M, K]
        B_shape = [batch_dim, K, N]
        D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype="float16", name="input_2", is_input=True)
        bmm_tensor = ops.gemm_universal.bmm_rrr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, input_2)
        add_tensor._attrs["name"] = "add_tensor"
        output = ops.elementwise(FuncEnum.ADD)(add_tensor, input_2)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            src_ops = list(tensor.src_ops())
            if len(src_ops) != 1:
                continue
            if src_ops[0]._attrs["op"].startswith("bmm"):
                check_tensor = tensor
                self.assertEqual(src_ops[0]._attrs["op"], "bmm_rrr_add")
                break
        self.assertIsNotNone(check_tensor)

        for B in Bs:
            X_pt = torch.randn(B, M, K).cuda().half()
            W_pt = torch.randn(B, K, N).cuda().half()
            D0_pt = torch.randn(B, M, N).cuda().half()
            Y_pt = torch.bmm(X_pt, W_pt) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = torch.empty([B, M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_rrr_add(self):
        self._test_bmm_rrr_add([8], 32, 16, 8, "bmm_rrr_add_basic")
        self._test_bmm_rrr_add([8, 32], 32, 16, 8, "bmm_rrr_add_dynamic")
        self._test_bmm_rrr_add([8], 7, 13, 3, "bmm_rrr_add_need_align")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
