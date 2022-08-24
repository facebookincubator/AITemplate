# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class GemmRrrSmallNkTestCase(unittest.TestCase):
    def _create_gemm_rrr_graph(self, M, K, N):
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[K, N], dtype="float16", name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "gemm_rrr_tensor"
        Y._attrs["is_output"] = True

        return X, W, Y

    def _test_small_nk(self, Ms, N, K, testname=None):
        if testname is None:
            testname = "gemm_rrr_small_nk_{}_{}_{}".format(Ms, N, K)
            testname = testname.replace(" ", "")
            testname = testname.replace("[", "")
            testname = testname.replace("]", "")

        X, W, gemm_tensor = self._create_gemm_rrr_graph(
            shape_utils.gen_int_var_min_max(Ms), K, N
        )

        output = ops.elementwise(FuncEnum.COS)(gemm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module([output, gemm_tensor], target, "./tmp", testname)

        output_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "gemm_rrr_tensor":
                output_tensor = tensor
                break

        self.assertIsNotNone(
            output_tensor, "Cannot find output tensor from module's graph"
        )
        self.assertEqual(
            len(output_tensor._attrs["src_ops"]),
            1,
            "Incorrect counts of src_ops in output",
        )

        src_op = list(output_tensor._attrs["src_ops"])[0]
        self.assertEqual(
            src_op._attrs["op"], "gemm_rrr_small_nk", "output op type incorrect"
        )

        for m in Ms:
            X_pt = torch.randn(m, K).cuda().half()
            W_pt = torch.randn(K, N).cuda().half()
            mm_pt = torch.matmul(X_pt, W_pt)
            Y_pt = torch.cos(mm_pt)
            y = torch.empty([m, N]).cuda().half()
            gemm_tensor_pt = torch.empty([m, N]).cuda().half()
            module.RunWithTensors(
                {"input_0": X_pt, "input_1": W_pt},
                {"output_0": y, "gemm_rrr_tensor": gemm_tensor_pt},
            )
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_small_nk(self):
        self._test_small_nk([10], 8, 4)
        self._test_small_nk([105], 8, 8)
        self._test_small_nk([1000], 6, 4)

    def test_small_nk_dynamic_shape(self):
        self._test_small_nk([10, 30], 6, 4, "dynamic")
        self._test_small_nk([10, 30, 50], 6, 4, "dynamic1")

    def test_small_nk_alignment(self):
        self._test_small_nk([1000], 6, 3)
        self._test_small_nk([10], 6, 3)
        self._test_small_nk([100, 200], 6, 3)
        self._test_small_nk([105], 7, 1)

    def test_small_nk_no_transform(self):
        M, K, N = 8, 8, 16
        _, _, output = self._create_gemm_rrr_graph(M, K, N)

        target = detect_target()
        module = gen_execution_module(
            output, target, "./tmp", "test_small_nk_fail_{}_{}_{}".format(M, K, N)
        )

        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "gemm_rrr_tensor":
                output_tensor = tensor
                break

        self.assertIsNotNone(
            output_tensor, "Cannot find output tensor from module's graph"
        )
        self.assertEqual(
            len(output_tensor._attrs["src_ops"]),
            1,
            "Incorrect counts of src_ops in output",
        )

        src_op = list(output_tensor._attrs["src_ops"])[0]
        self.assertEqual(src_op._attrs["op"], "gemm_rrr", "output op type incorrect")

        X_pt = torch.randn(M, K).cuda().half()
        W_pt = torch.randn(K, N).cuda().half()
        Y_pt = torch.matmul(X_pt, W_pt)

        y = torch.empty([M, N]).cuda().half()
        module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


class BmmRcrN1TestCase(unittest.TestCase):
    def _create_bmm_rcr_graph(self, B, M, N, K):
        X = Tensor(shape=[B, M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "bmm_rcr_tensor"

        return X, W, Y

    def _test_n1_k8(self, B, M, N, K, testname=None):
        if testname is None:
            testname = "bmm_rcr_n1_{}_{}_{}_{}".format(B, M, N, K)
            testname = testname.replace(" ", "")
            testname = testname.replace("[", "")
            testname = testname.replace("]", "")

        X, W, bmm_tensor = self._create_bmm_rcr_graph(
            B, shape_utils.gen_int_var_min_max(M), N, K
        )
        mul = ops.elementwise(FuncEnum.MUL)(bmm_tensor, Tensor(shape=[], value=1.0))
        output = ops.elementwise(FuncEnum.COS)(mul)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(output, target, "./tmp", testname)

        output_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_rcr_tensor":
                output_tensor = tensor
                break

        assert output_tensor is not None
        assert len(output_tensor._attrs["src_ops"]) == 1
        src_op = list(output_tensor._attrs["src_ops"])[0]
        assert src_op._attrs["op"] == "bmm_rcr_n1"

        for m in M:
            X_pt = torch.randn(B, m, K).cuda().half()
            W_pt = torch.randn(B, N, K).cuda().half()

            def pt_bmm(X_pt, W_pt):
                WT = torch.transpose(W_pt, 2, 1)
                Y_pt = torch.bmm(X_pt, WT)
                return Y_pt

            Y_pt = torch.cos(pt_bmm(X_pt, W_pt))

            y = torch.empty([B, m, N]).cuda().half()
            module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_n1_k8(self):
        self._test_n1_k8(1, [8], 1, 8)
        self._test_n1_k8(10, [8], 1, 8)

    def test_n1_k8_dynamic(self):
        self._test_n1_k8(10, [8, 16], 1, 8)

    def test_n_non1_fail(self):
        B, M, K, N = 8, 8, 8, 8
        _, _, output = self._create_bmm_rcr_graph(B, M, K, N)
        output._attrs["is_output"] = True

        target = detect_target()
        module = gen_execution_module(output, target, "./tmp", "bmm_rcr_n_non1")

        output_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_rcr_tensor":
                output_tensor = tensor
                break

        self.assertIsNotNone(output_tensor, "bmm_rcr tensor not found")
        self.assertEqual(len(output_tensor._attrs["src_ops"]), 1)
        src_op = next(iter(output_tensor._attrs["src_ops"]))
        self.assertEqual(src_op._attrs["op"], "bmm_rcr")


if __name__ == "__main__":
    unittest.main()
