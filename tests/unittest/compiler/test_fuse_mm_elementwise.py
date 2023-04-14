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
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import shape_utils

from parameterized import parameterized


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s_sm80" % (
        testcase_func.__name__[:-5],
        param.args[-2],
    )


class FuseGemmRcrBiasCase(unittest.TestCase):
    def _build_gemm_rcr_bias(self, M, N, K, decomposed, dtype):
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)

        if decomposed:
            gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
            bias_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        else:
            bias_tensor = ops.gemm_universal.gemm_rcr_bias()(input_0, input_1, input_2)

        return bias_tensor

    def _build_gemm_rcr_bias_add_add_relu_chain(
        self, M, N, K, depth, decomposed, dtype
    ):
        D_shape = [M, N]
        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)
        input_4 = Tensor(shape=D_shape, dtype=dtype, name="input_4", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed, dtype)
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

    def _build_gemm_rcr_bias_mul(self, M, N, K, decomposed, dtype):
        D_shape = [M, N]
        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed, dtype)
        mul_tensor = ops.elementwise(FuncEnum.MUL)(bias_tensor, input_3)

        return mul_tensor

    def _test_gemm_rcr_bias(self, Ms, N, K, decomposed, testname, dtype="float16"):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        bias_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 1, decomposed, dtype
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            y = get_torch_empty_tensor([M, N], dtype)

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add(self, Ms, N, K, decomposed, testname, dtype="float16"):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        add_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 2, decomposed, dtype
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + D0_pt)

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt
            y = get_torch_empty_tensor([M, N], dtype)

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_add(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        add2_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 3, decomposed, dtype
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            D1_pt = get_random_torch_tensor([M, N], dtype)
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

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_add_relu(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        relu_tensor = self._build_gemm_rcr_bias_add_add_relu_chain(
            m_dim, N, K, 4, decomposed, dtype
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            D1_pt = get_random_torch_tensor([M, N], dtype)
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

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_gemm_rcr_bias_add_fail(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M, N, K = 16, 32, 8
        B_shape = [N]

        input_3 = Tensor(shape=B_shape, dtype=dtype, name="input_3", is_input=True)

        gemm_bias_tensor = self._build_gemm_rcr_bias(M, N, K, False, dtype)
        gemm_bias_tensor._attrs["name"] = "gemm_tensor"
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_bias_tensor, input_3)
        add_tensor._attrs["name"] = "gemm_bias_add_tensor"

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        module = compile_model(
            output, target, "./tmp", f"gemm_bias_fusion_add_fail_{dtype}"
        )

        # This shouldn't be merged into gemm_rcr_bias_add since input_3 needs broadcasting
        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "gemm_tensor":
                check_tensor = tensor
        self.assertIsNotNone(check_tensor)
        self.assertEqual(len(check_tensor.src_ops()), 1)
        src_op = list(check_tensor.src_ops())[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rcr_bias")

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        B1_pt = get_random_torch_tensor([N], dtype)
        Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) + B1_pt)

        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors([X_pt, W_pt, B_pt, B1_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_gemm_rcr_bias_chained(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M, N, K = 16, 32, 8
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)

        gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        add_tensor._attrs["name"] = "first_gemm"

        D_shape = [N, N]
        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)
        gemm1_tensor = ops.gemm_universal.gemm_rcr()(add_tensor, input_3)
        add1_tensor = ops.elementwise(FuncEnum.ADD)(gemm1_tensor, input_2)
        add1_tensor._attrs["name"] = "second_gemm"

        output = ops.elementwise(FuncEnum.COS)(add1_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        module = compile_model(
            output, target, "./tmp", f"gemm_bias_fusion_chained_{dtype}"
        )

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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        D_pt = get_random_torch_tensor([N, N], dtype)
        Y_pt = torch.cos(
            torch.nn.functional.linear(
                torch.nn.functional.linear(X_pt, W_pt, bias=B_pt), D_pt, bias=B_pt
            )
        )

        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors([X_pt, W_pt, B_pt, D_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_gemm_rcr_bias_fail(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M, N, K = 16, 32, 8
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [M, N]

        input_0 = Tensor(shape=X_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)

        gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
        add_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        add_tensor._attrs["name"] = "final_tensor"

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        module = compile_model(
            output, target, "./tmp", f"gemm_bias_fusion_fail_{dtype}"
        )

        check_tensor = None
        for tensor in module.debug_sorted_graph:
            if len(tensor.src_ops()) != 1:
                continue
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"] == "gemm_rcr":
                check_tensor = tensor
                break
        self.assertIsNotNone(check_tensor)

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([M, N], dtype)
        Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt) + B_pt)

        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors([X_pt, W_pt, B_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_add_relu(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]

        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
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

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_tanh(self, Ms, N, K, decomposed, testname, dtype="float16"):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            Y_pt = torch.cos(
                torch.tanh(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt))
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul(self, Ms, N, K, decomposed, testname, dtype="float16"):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            Y_pt = torch.cos(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) * D0_pt)

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 4
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul_add(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]

        input_4 = Tensor(shape=D_shape, dtype=dtype, name="input_4", is_input=True)
        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            D1_pt = get_random_torch_tensor([M, N], dtype)
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

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_mul_tanh(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")

        mul_tensor = self._build_gemm_rcr_bias_mul(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D0_pt = get_random_torch_tensor([M, N], dtype)
            Y_pt = torch.cos(
                torch.tanh(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt) * D0_pt)
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 4
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D0_pt

            y = get_torch_empty_tensor([M, N], dtype)
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

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @parameterized.expand(
        [
            (
                _test_gemm_rcr_bias,
                [8],
                16,
                8,
                True,
                "gemm_rcr_bias_basic_decomposed_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_add_add,
                [8],
                16,
                8,
                False,
                "gemm_rcr_bias_add_basic_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_add_add,
                [8, 32],
                16,
                8,
                False,
                "gemm_rcr_bias_add_add_dynamic_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_add_add_relu,
                [8],
                16,
                3,
                False,
                "gemm_rcr_bias_add_add_relu_need_align_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_add_relu,
                [8],
                16,
                8,
                True,
                "gemm_rcr_bias_add_relu_basic_decomposed_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_tanh,
                [8],
                16,
                8,
                False,
                "gemm_rcr_bias_tanh_basic_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_mul,
                [8, 32],
                16,
                8,
                False,
                "gemm_rcr_bias_mul_dynamic_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_mul_add,
                [8],
                16,
                3,
                False,
                "gemm_rcr_bias_mul_add_need_align_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_mul_tanh,
                [8],
                16,
                3,
                False,
                "gemm_rcr_bias_mul_tanh_need_align_float",
                "float",
            ),
        ],
        name_func=custom_name_func,
    )
    def test_gemm_rcr_bias_add_float_sm80(
        self, func, Ms, N, K, decomposed, testname, dtype
    ):
        func(self, Ms, N, K, decomposed, testname, dtype)


filter_test_cases_by_test_env(FuseGemmRcrBiasCase)


class FuseGemmRcrBiasActivationCase(unittest.TestCase):
    def _build_gemm_rcr_bias(self, M, N, K, decomposed, dtype):
        X_shape = [M, K]
        W_shape = [N, K]
        B_shape = [N]

        input_0 = Tensor(shape=X_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=W_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)

        if decomposed:
            gemm_tensor = ops.gemm_universal.gemm_rcr()(input_0, input_1)
            bias_tensor = ops.elementwise(FuncEnum.ADD)(gemm_tensor, input_2)
        else:
            bias_tensor = ops.gemm_rcr_bias()(input_0, input_1, input_2)

        return bias_tensor

    def _build_gemm_rcr_bias_sigmoid(self, M, N, K, decomposed, dtype):
        gemm_tensor = self._build_gemm_rcr_bias(M, N, K, decomposed, dtype)
        sigmoid_tensor = ops.elementwise(FuncEnum.SIGMOID)(gemm_tensor)

        return sigmoid_tensor

    def _test_gemm_rcr_bias_activation(
        self, Ms, N, K, activation, target_ait, decomposed, testname, dtype="float16"
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
        elif activation == "gelu":
            ait_func = FuncEnum.GELU
            pt_func = torch.nn.functional.gelu
        elif activation == "fast_gelu":
            ait_func = FuncEnum.FASTGELU
            pt_func = torch.nn.functional.gelu
        else:
            raise AssertionError("Activation not supported")

        bias_tensor = self._build_gemm_rcr_bias(m_dim, N, K, decomposed, dtype)
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            Y_pt = torch.cos(pt_func(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)))

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            y = get_torch_empty_tensor([M, N], dtype)

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_sigmoid_mul(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]
        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)

        sigmoid_tensor = self._build_gemm_rcr_bias_sigmoid(
            m_dim, N, K, decomposed, dtype
        )
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D_pt = get_random_torch_tensor([M, N], dtype)
            Y_pt = torch.cos(
                torch.sigmoid(torch.nn.functional.linear(X_pt, W_pt, B_pt)) * D_pt
            )

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D_pt

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_gemm_rcr_bias_sigmoid_mul_tanh(
        self, Ms, N, K, decomposed, testname, dtype="float16"
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        D_shape = [m_dim, N]
        input_3 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)

        sigmoid_tensor = self._build_gemm_rcr_bias_sigmoid(
            m_dim, N, K, decomposed, dtype
        )
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D_pt = get_random_torch_tensor([M, N], dtype)
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

            y = get_torch_empty_tensor([M, N], dtype)
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

    def test_gemm_rcr_bias_gelu(self):
        self._test_gemm_rcr_bias_activation(
            [8, 32],
            16,
            8,
            "gelu",
            "gemm_rcr_bias_gelu",
            True,
            "gemm_rcr_bias_gelu_basic_decomposed",
        )
        self._test_gemm_rcr_bias_activation(
            [8, 32],
            16,
            8,
            "fast_gelu",
            "gemm_rcr_bias_fast_gelu",
            True,
            "gemm_rcr_bias_fast_gelu_basic_decomposed",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @parameterized.expand(
        [
            (
                _test_gemm_rcr_bias_activation,
                [8],
                16,
                8,
                True,
                "gemm_rcr_bias_relu_basic_decomposed_float",
                "float",
                "relu",
                "gemm_rcr_bias_relu",
            ),
            (
                _test_gemm_rcr_bias_activation,
                [8],
                16,
                8,
                False,
                "gemm_rcr_bias_sigmoid_basic_float",
                "float",
                "sigmoid",
                "gemm_rcr_bias_sigmoid",
            ),
            (
                _test_gemm_rcr_bias_sigmoid_mul,
                [8],
                16,
                8,
                False,
                "gemm_rcr_bias_sigmoid_mul_basic_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_sigmoid_mul_tanh,
                [8],
                16,
                3,
                False,
                "gemm_rcr_bias_sigmoid_mul_tanh_need_align_float",
                "float",
            ),
            (
                _test_gemm_rcr_bias_activation,
                [8],
                16,
                8,
                False,
                "gemm_rcr_bias_tanh_basic_float",
                "float",
                "tanh",
                "gemm_rcr_bias_tanh",
            ),
            (
                _test_gemm_rcr_bias_activation,
                [8, 32],
                16,
                8,
                True,
                "gemm_rcr_bias_fast_gelu_basic_decomposed_float",
                "float",
                "fast_gelu",
                "gemm_rcr_bias_fast_gelu",
            ),
        ],
        name_func=custom_name_func,
    )
    def test_gemm_rcr_bias_float_sm80(
        self,
        func,
        Ms,
        N,
        K,
        decomposed,
        testname,
        dtype,
        activation=None,
        target_ait=None,
    ):
        if activation and target_ait:
            func(self, Ms, N, K, activation, target_ait, decomposed, testname, dtype)
        else:
            func(self, Ms, N, K, decomposed, testname, dtype)


filter_test_cases_by_test_env(FuseGemmRcrBiasActivationCase)


class FuseGemmRcrBiasSwishCase(unittest.TestCase):
    def _test_gemm_rcr_bias_swish(
        self, Ms, N, K, testname, dtype="float16", use_add=False, use_silu=False
    ):
        m_dim = shape_utils.gen_int_var_min_max(Ms, name="M_size")
        X_shape = [m_dim, K]
        W_shape = [N, K]
        B_shape = [N]
        D_shape = [m_dim, N]
        input_1 = Tensor(shape=X_shape, dtype=dtype, name="input_0", is_input=True)
        input_2 = Tensor(shape=W_shape, dtype=dtype, name="input_1", is_input=True)
        input_3 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)
        input_4 = Tensor(shape=D_shape, dtype=dtype, name="input_3", is_input=True)

        if use_add:
            tensor = ops.gemm_rcr()(input_1, input_2)
            gemm_tensor = ops.elementwise(FuncEnum.ADD)(tensor, input_3)
        else:
            gemm_tensor = ops.gemm_rcr_bias()(input_1, input_2, input_3)

        if use_silu:
            swish_tensor = ops.elementwise(FuncEnum.SILU)(gemm_tensor)
        else:
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
            X_pt = get_random_torch_tensor([M, K], dtype)
            W_pt = get_random_torch_tensor([N, K], dtype)
            B_pt = get_random_torch_tensor([N], dtype)
            D_pt = get_random_torch_tensor([M, N], dtype)
            gemm_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
            Y_pt = gemm_pt * torch.sigmoid(gemm_pt) + D_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = B_pt
            inputs[input_name_to_index["input_3"]] = D_pt

            y = get_torch_empty_tensor([M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_gemm_rcr_bias_swish(self):
        self._test_gemm_rcr_bias_swish([8], 16, 8, "gemm_rcr_bias_swish_basic")
        self._test_gemm_rcr_bias_swish([8, 32], 16, 8, "gemm_rcr_bias_swish_dynamic")
        self._test_gemm_rcr_bias_swish([8], 16, 3, "gemm_rcr_bias_swish_need_align")
        self._test_gemm_rcr_bias_swish(
            [8], 16, 3, "gemm_rcr_bias_silu_basic", use_silu=True
        )

    def test_gemm_rcr_add_swish(self):
        self._test_gemm_rcr_bias_swish(
            [8], 16, 8, "gemm_rcr_add_swish_basic", use_add=True
        )
        self._test_gemm_rcr_bias_swish(
            [8, 32], 16, 8, "gemm_rcr_add_swish_dynamic", use_add=True
        )
        self._test_gemm_rcr_bias_swish(
            [8], 16, 3, "gemm_rcr_add_swish_need_align", use_add=True
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_gemm_rcr_swish_float_sm80(self):
        self._test_gemm_rcr_bias_swish(
            [8],
            16,
            8,
            "gemm_rcr_bias_swish_basic_float",
            dtype="float",
        )
        self._test_gemm_rcr_bias_swish(
            [8, 32],
            16,
            8,
            "gemm_rcr_add_swish_dynamic_float",
            dtype="float",
            use_add=True,
        )
        self._test_gemm_rcr_bias_swish(
            [8],
            16,
            3,
            "gemm_rcr_add_swish_need_align_float",
            dtype="float",
            use_add=True,
        )


filter_test_cases_by_test_env(FuseGemmRcrBiasSwishCase)


class FuseBmmCcrAddCase(unittest.TestCase):
    def _test_bmm_ccr_add(
        self, Bs, M, N, K, testname, dtype="float16", do_not_fuse=False
    ):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, K, M]
        B_shape = [batch_dim, N, K]
        if do_not_fuse:
            assert M != 1
            D0_shape = [batch_dim, 1, N]
        else:
            D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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
                if do_not_fuse:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_ccr")
                else:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_ccr_add")
                break
        self.assertIsNotNone(check_tensor)

        if do_not_fuse:
            return

        for B in Bs:
            X_pt = get_random_torch_tensor([B, K, M], dtype)
            W_pt = get_random_torch_tensor([B, N, K], dtype)
            D0_pt = get_random_torch_tensor([B, M, N], dtype)
            Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt.transpose(2, 1)) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = get_torch_empty_tensor([B, M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_bmm_ccr_add_negative(self, testname, negative_type, dtype="float16"):
        B, K, M, N = 8, 32, 16, 8
        A_shape = [B, K, M]
        B_shape = [B, N, K]
        D0_shape = [B, M, N]
        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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

        X_pt = get_random_torch_tensor([B, K, M], dtype)
        W_pt = get_random_torch_tensor([B, N, K], dtype)
        D0_pt = get_random_torch_tensor([B, M, N], dtype)

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

        y = get_torch_empty_tensor([B, M, N], dtype)
        y1 = get_torch_empty_tensor([B, M, N], dtype)
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
        self._test_bmm_ccr_add(
            [8], 32, 16, 8, "bmm_ccr_add_do_not_fuse", do_not_fuse=True
        )

    def test_bmm_ccr_add_negative(self):
        self._test_bmm_ccr_add_negative("bmm_ccr_add_negative_output", "is_output")
        self._test_bmm_ccr_add_negative("bmm_ccr_add_negative_input", "other_input")

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_bmm_ccr_add_float_sm80(self):
        self._test_bmm_ccr_add(
            [8, 32], 32, 16, 8, "bmm_ccr_add_dynamic_float", dtype="float"
        )
        self._test_bmm_ccr_add(
            [8], 7, 13, 3, "bmm_ccr_add_need_align_float", dtype="float"
        )
        self._test_bmm_ccr_add(
            [8],
            32,
            16,
            8,
            "bmm_ccr_add_do_not_fuse_float",
            dtype="float",
            do_not_fuse=True,
        )
        self._test_bmm_ccr_add_negative(
            "bmm_ccr_add_negative_output", "is_output", dtype="float"
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_bmm_ccr_add_double_shared_input(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        B, M, N, K = 8, 32, 16, 8

        A_shape = [B, K, M]
        B_shape = [B, N, K]
        D0_shape = [B, M, N]
        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_11 = Tensor(shape=B_shape, dtype=dtype, name="input_11", is_input=True)
        bmm_tensor = ops.gemm_universal.bmm_ccr()(input_0, input_1)
        bmm_tensor_1 = ops.gemm_universal.bmm_ccr()(input_0, input_11)

        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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
        module = compile_model(
            [output, output_1], target, "./tmp", f"bmm_ccr_double_shared_inputs_{dtype}"
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

        X_pt = get_random_torch_tensor([B, K, M], dtype)
        W_pt = get_random_torch_tensor([B, N, K], dtype)
        W1_pt = get_random_torch_tensor([B, N, K], dtype)
        D0_pt = get_random_torch_tensor([B, M, N], dtype)
        Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt.transpose(2, 1)) + D0_pt + D0_pt
        Y1_pt = torch.bmm(X_pt.transpose(2, 1), W1_pt.transpose(2, 1)) + D0_pt + D0_pt

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [None] * 4
        inputs[input_name_to_index["input_0"]] = X_pt
        inputs[input_name_to_index["input_1"]] = W_pt
        inputs[input_name_to_index["input_11"]] = W1_pt
        inputs[input_name_to_index["input_2"]] = D0_pt

        y = get_torch_empty_tensor([B, M, N], dtype)
        y1 = get_torch_empty_tensor([B, M, N], dtype)
        ys = [None] * 2
        output_name_to_index = module.get_output_name_to_index_map()
        ys[output_name_to_index["output_0"]] = y
        ys[output_name_to_index["output_1"]] = y1

        module.run_with_tensors(inputs, ys)

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))


filter_test_cases_by_test_env(FuseBmmCcrAddCase)


class FuseBmmCrrAddCase(unittest.TestCase):
    def _test_bmm_crr_add(
        self, Bs, M, N, K, testname, dtype="float16", do_not_fuse=False
    ):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, K, M]
        B_shape = [batch_dim, K, N]
        if do_not_fuse:
            assert M != 1
            D0_shape = [batch_dim, 1, N]
        else:
            D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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
                if do_not_fuse:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_crr")
                else:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_crr_add")
                break
        self.assertIsNotNone(check_tensor)

        if do_not_fuse:
            return

        for B in Bs:
            X_pt = get_random_torch_tensor([B, K, M], dtype)
            W_pt = get_random_torch_tensor([B, K, N], dtype)
            D0_pt = get_random_torch_tensor([B, M, N], dtype)
            Y_pt = torch.bmm(X_pt.transpose(2, 1), W_pt) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = get_torch_empty_tensor([B, M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_crr_add(self):
        self._test_bmm_crr_add([8], 32, 16, 8, "bmm_crr_add_basic")
        self._test_bmm_crr_add([8, 32], 32, 16, 8, "bmm_crr_add_dynamic")
        self._test_bmm_crr_add([8], 7, 13, 3, "bmm_crr_add_need_align")
        self._test_bmm_crr_add(
            [8], 32, 16, 8, "bmm_crr_add_do_not_fuse", do_not_fuse=True
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_bmm_crr_add_float_sm80(self):
        self._test_bmm_crr_add(
            [8, 32], 32, 16, 8, "bmm_crr_add_dynamic_float", dtype="float"
        )
        self._test_bmm_crr_add(
            [8], 7, 13, 3, "bmm_crr_add_need_align_float", dtype="float"
        )
        self._test_bmm_crr_add(
            [8], 32, 16, 8, "bmm_crr_add_do_not_fuse", dtype="float", do_not_fuse=True
        )


filter_test_cases_by_test_env(FuseBmmCrrAddCase)


class FuseBmmRrrAddCase(unittest.TestCase):
    def _test_bmm_rrr_add(
        self, Bs, M, N, K, testname, dtype="float16", do_not_fuse=False
    ):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, M, K]
        B_shape = [batch_dim, K, N]
        if do_not_fuse:
            assert M != 1
            D0_shape = [batch_dim, 1, N]
        else:
            D0_shape = [batch_dim, M, N]
        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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
                if do_not_fuse:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_rrr")
                else:
                    self.assertEqual(src_ops[0]._attrs["op"], "bmm_rrr_add")
                break
        self.assertIsNotNone(check_tensor)

        if do_not_fuse:
            return

        for B in Bs:
            X_pt = get_random_torch_tensor([B, M, K], dtype)
            W_pt = get_random_torch_tensor([B, K, N], dtype)
            D0_pt = get_random_torch_tensor([B, M, N], dtype)
            Y_pt = torch.bmm(X_pt, W_pt) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = get_torch_empty_tensor([B, M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_rrr_add(self):
        self._test_bmm_rrr_add([8], 32, 16, 8, "bmm_rrr_add_basic")
        self._test_bmm_rrr_add([8, 32], 32, 16, 8, "bmm_rrr_add_dynamic")
        self._test_bmm_rrr_add([8], 7, 13, 3, "bmm_rrr_add_need_align")
        self._test_bmm_rrr_add([8], 32, 16, 8, "bmm_rrr_add_no_fuse", do_not_fuse=True)

    def _test_bmm_rrr_bias_add(
        self, Bs, M, N, K, bias_shapes, testname, dtype="float16"
    ):
        batch_dim = shape_utils.gen_int_var_min_max(Bs, name="batch_size")
        A_shape = [batch_dim, M, K]
        B_shape = [batch_dim, K, N]
        D0_shape = bias_shapes

        input_0 = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        input_1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        input_2 = Tensor(shape=D0_shape, dtype=dtype, name="input_2", is_input=True)
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
        self.assertIsNotNone(check_tensor)

        for B in Bs:
            X_pt = get_random_torch_tensor([B, M, K], dtype)
            W_pt = get_random_torch_tensor([B, K, N], dtype)
            D0_pt = get_random_torch_tensor(D0_shape, dtype)
            Y_pt = torch.bmm(X_pt, W_pt) + D0_pt + D0_pt

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [None] * 3
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            inputs[input_name_to_index["input_2"]] = D0_pt

            y = get_torch_empty_tensor([B, M, N], dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_rrr_bias_add(self):
        self._test_bmm_rrr_bias_add([8], 32, 16, 8, [16], "bmm_rrr_bias_add_01")
        self._test_bmm_rrr_bias_add([8], 32, 16, 8, [32, 16], "bmm_rrr_bias_add_02")
        self._test_bmm_rrr_bias_add([8], 32, 16, 8, [1, 32, 16], "bmm_rrr_bias_add_03")
        self._test_bmm_rrr_bias_add([8], 32, 16, 8, [1, 16], "bmm_rrr_bias_add_03")

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_bmm_rrr_add_float_sm80(self):
        self._test_bmm_rrr_add(
            [8, 32], 32, 16, 8, "bmm_rrr_add_dynamic_float", dtype="float"
        )
        self._test_bmm_rrr_add(
            [8], 7, 13, 3, "bmm_rrr_add_need_align_float", dtype="float"
        )
        self._test_bmm_rrr_add(
            [8], 32, 16, 8, "bmm_rrr_add_no_fuse_float", dtype="float", do_not_fuse=True
        )
        self._test_bmm_rrr_bias_add(
            [8], 32, 16, 8, [1, 32, 16], "bmm_rrr_bias_add_float_03", dtype="float"
        )


filter_test_cases_by_test_env(FuseBmmRrrAddCase)

if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
