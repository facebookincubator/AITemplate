# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import logger


class StridedSplitGroupGemmTestCase(unittest.TestCase):
    def test_split_group_gemm(self):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(
            shape=[IntImm(M), IntImm(K)],
            dtype="float16",
            name="x",
            is_input=True,
        )
        W1 = Tensor(shape=[N, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype="float16", name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype="float16", name="w3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr()
        Y = group_gemm_op(
            operand_groups=[[X1, W1], [X2, W2], [X3, W3]], output_stride_dim=dim
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_cat.so"
        module = gen_execution_module(
            [Y], target, "./tmp", "strided_split_group_gemm_rcr_cat", dll_name=dll_name
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 1)
        Y_src_op = Y_src_ops[0]
        np.testing.assert_equal(Y_src_op, group_gemm_op)
        expected_inputs_group_gemm_op = [X, W1, X, W2, X, W3]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X_pt = torch.randn(M, K).cuda().half()
        W1_pt = torch.randn(N, K1).cuda().half()
        W2_pt = torch.randn(N, K2).cuda().half()
        W3_pt = torch.randn(N, K3).cuda().half()
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        inputs = [
            X_pt,
            W1_pt,
            W2_pt,
            W3_pt,
        ]
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_split_group_gemm_bias(self):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(
            shape=[IntImm(M), IntImm(K)], dtype="float16", name="x", is_input=True
        )
        W1 = Tensor(shape=[N, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype="float16", name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype="float16", name="w3", is_input=True)
        B1 = Tensor(shape=[N], dtype="float16", name="b1", is_input=True)
        B2 = Tensor(shape=[N], dtype="float16", name="b2", is_input=True)
        B3 = Tensor(shape=[N], dtype="float16", name="b3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr_bias()
        Y = group_gemm_op(
            operand_groups=[[X1, W1, B1], [X2, W2, B2], [X3, W3, B3]],
            output_stride_dim=dim,
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_bias_cat.so"
        module = gen_execution_module(
            [Y],
            target,
            "./tmp",
            "strided_split_group_gemm_rcr_bias_cat",
            dll_name=dll_name,
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 1)
        Y_src_op = Y_src_ops[0]
        np.testing.assert_equal(Y_src_op, group_gemm_op)
        expected_inputs_group_gemm_op = [X, W1, B1, X, W2, B2, X, W3, B3]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X_pt = torch.randn(M, K).cuda().half()
        W1_pt = torch.randn(N, K1).cuda().half()
        W2_pt = torch.randn(N, K2).cuda().half()
        W3_pt = torch.randn(N, K3).cuda().half()
        B1_pt = torch.randn(N).cuda().half()
        B2_pt = torch.randn(N).cuda().half()
        B3_pt = torch.randn(N).cuda().half()
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt, bias=B3_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        input_name_to_index = module.GetInputNameToIndexMap()
        inputs = [0] * 7
        inputs[input_name_to_index["x"]] = X_pt
        inputs[input_name_to_index["w1"]] = W1_pt
        inputs[input_name_to_index["w2"]] = W2_pt
        inputs[input_name_to_index["w3"]] = W3_pt
        inputs[input_name_to_index["b1"]] = B1_pt
        inputs[input_name_to_index["b2"]] = B2_pt
        inputs[input_name_to_index["b3"]] = B3_pt
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_split_group_gemm_reorder(self):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(
            shape=[IntImm(M), IntImm(K)], dtype="float16", name="x", is_input=True
        )
        W1 = Tensor(shape=[N, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype="float16", name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype="float16", name="w3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr()
        Y = group_gemm_op(
            operand_groups=[[X2, W2], [X1, W1], [X3, W3]], output_stride_dim=dim
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_cat_reorder.so"
        module = gen_execution_module(
            [Y], target, "./tmp", "strided_split_group_gemm_rcr_cat", dll_name=dll_name
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 1)
        Y_src_op = Y_src_ops[0]
        np.testing.assert_equal(Y_src_op, group_gemm_op)
        expected_inputs_group_gemm_op = [X, W2, X, W1, X, W3]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X_pt = torch.randn(M, K).cuda().half()
        W1_pt = torch.randn(N, K1).cuda().half()
        W2_pt = torch.randn(N, K2).cuda().half()
        W3_pt = torch.randn(N, K3).cuda().half()
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt)
        Y_pt = torch.cat([Y2_pt, Y1_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        inputs = [0 for i in range(4)]
        name_to_idx = module.GetInputNameToIndexMap()
        inputs[name_to_idx["x"]] = X_pt
        inputs[name_to_idx["w1"]] = W1_pt
        inputs[name_to_idx["w2"]] = W2_pt
        inputs[name_to_idx["w3"]] = W3_pt
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_split_group_gemm_bias_reorder(self):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(
            shape=[IntImm(M), IntImm(K)], dtype="float16", name="x", is_input=True
        )
        W1 = Tensor(shape=[N, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype="float16", name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype="float16", name="w3", is_input=True)
        B1 = Tensor(shape=[N], dtype="float16", name="b1", is_input=True)
        B2 = Tensor(shape=[N], dtype="float16", name="b2", is_input=True)
        B3 = Tensor(shape=[N], dtype="float16", name="b3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr_bias()
        Y = group_gemm_op(
            operand_groups=[[X2, W2, B2], [X3, W3, B3], [X1, W1, B1]],
            output_stride_dim=dim,
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_bias_cat_reorder.so"
        module = gen_execution_module(
            [Y],
            target,
            "./tmp",
            "strided_split_group_gemm_rcr_bias_cat",
            dll_name=dll_name,
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 1)
        Y_src_op = Y_src_ops[0]
        np.testing.assert_equal(Y_src_op, group_gemm_op)
        expected_inputs_group_gemm_op = [X, W2, B2, X, W3, B3, X, W1, B1]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X_pt = torch.randn(M, K).cuda().half()
        W1_pt = torch.randn(N, K1).cuda().half()
        W2_pt = torch.randn(N, K2).cuda().half()
        W3_pt = torch.randn(N, K3).cuda().half()
        B1_pt = torch.randn(N).cuda().half()
        B2_pt = torch.randn(N).cuda().half()
        B3_pt = torch.randn(N).cuda().half()
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt, bias=B3_pt)
        Y_pt = torch.cat([Y2_pt, Y3_pt, Y1_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        input_name_to_index = module.GetInputNameToIndexMap()
        inputs = [0] * 7
        inputs[input_name_to_index["x"]] = X_pt
        inputs[input_name_to_index["w1"]] = W1_pt
        inputs[input_name_to_index["w2"]] = W2_pt
        inputs[input_name_to_index["w3"]] = W3_pt
        inputs[input_name_to_index["b1"]] = B1_pt
        inputs[input_name_to_index["b2"]] = B2_pt
        inputs[input_name_to_index["b3"]] = B3_pt
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
