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
import logging
import unittest

import numpy as np
import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import logger


class StridedReshapeCatTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StridedReshapeCatTestCase, self).__init__(*args, **kwargs)
        self.test_count = 1

    def _test_strided_reshape_cat(self, num_cat_ops=1):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        M1 = 128
        N1 = 32
        K1 = 32

        M2 = 128
        N2 = 8
        K2 = 16

        M3 = 128
        N3 = 16
        K3 = 16

        BS = 128
        Input_M = 2
        Input_N = 8

        dim = 1

        X1 = Tensor(
            shape=[IntImm(M1), IntImm(K1)], dtype="float16", name="x1", is_input=True
        )
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        X2 = Tensor(
            shape=[IntImm(M2), IntImm(K2)], dtype="float16", name="x2", is_input=True
        )
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)

        X3 = Tensor(
            shape=[IntImm(M3), IntImm(K3)], dtype="float16", name="x3", is_input=True
        )
        W3 = Tensor(shape=[N3, K3], dtype="float16", name="w3", is_input=True)

        Input = Tensor(
            shape=[BS, Input_M, Input_N], dtype="float16", name="input", is_input=True
        )

        group_gemm_op = ops.group_gemm_rcr()
        Y1_orig, Y2_orig, Y3_orig = group_gemm_op(
            operand_groups=[[X1, W1], [X2, W2], [X3, W3]]
        )
        Y1 = ops.reshape()(Y1_orig, [BS, -1, Input_N])
        Y2 = ops.unsqueeze(dim)(Y2_orig)
        Y3 = ops.reshape()(Y3_orig, [BS, -1, Input_N])
        Y1._attrs["name"] = "y1"
        Y2._attrs["name"] = "y2"
        Y3._attrs["name"] = "y3"
        if num_cat_ops == 1:
            concat_op = ops.concatenate()
            Y = concat_op([Y1, Y2, Input, Y3], dim=dim)
        else:
            concat_op_1 = ops.concatenate()
            concat_op_2 = ops.concatenate()
            Y4 = concat_op_1([Y1, Y2], dim=dim)
            Y5 = concat_op_2([Input, Y3], dim=dim)
            Y6 = ops.reshape()(Y4, [BS, -1, Input_N])
            Y7 = ops.reshape()(Y5, [BS, -1, Input_N])
            Y = ops.concatenate()([Y6, Y7], dim=dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            [Y], target, "./tmp", "strided_reshape_cat", dll_name=dll_name
        )

        Y_src_ops = Y._attrs["src_ops"]
        if num_cat_ops == 1:
            np.testing.assert_equal(len(Y_src_ops), 2)
            np.testing.assert_equal(Y_src_ops, {group_gemm_op, concat_op})
            np.testing.assert_equal(
                concat_op._attrs["input_masks"], [False, False, True, False]
            )
        else:
            np.testing.assert_equal(concat_op_1._attrs["input_masks"], [False, False])
            np.testing.assert_equal(concat_op_2._attrs["input_masks"], [True, False])

        expected_inputs_group_gemm_op = [X1, W1, X2, W2, X3, W3]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X1_pt = torch.randn(M1, K1).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        X2_pt = torch.randn(M2, K2).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        X3_pt = torch.randn(M3, K3).cuda().half()
        W3_pt = torch.randn(N3, K3).cuda().half()
        Input_pt = torch.randn(BS, Input_M, Input_N).cuda().half()
        Y1_orig_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_orig_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y3_orig_pt = torch.nn.functional.linear(X3_pt, W3_pt)
        Y1_pt = torch.reshape(Y1_orig_pt, [BS, -1, Input_N])
        Y2_pt = torch.unsqueeze(Y2_orig_pt, dim)
        Y3_pt = torch.reshape(Y3_orig_pt, [BS, -1, Input_N])
        Y_pt = torch.cat([Y1_pt, Y2_pt, Input_pt, Y3_pt], dim=dim)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "x3": X3_pt,
            "w3": W3_pt,
            "input": Input_pt,
        }

        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.test_count += 1

    def test_strided_reshape_cat(self, num_cat_ops=1):
        self._test_strided_reshape_cat(1)
        self._test_strided_reshape_cat(2)

    def test_strided_reshape_cat_bias(self):
        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Group Gemm need SM80 HW")
            return

        M1 = 128
        N1 = 32
        K1 = 32

        M2 = 128
        N2 = 8
        K2 = 16

        BS = 128
        Input_M = 2
        Input_N = 8

        dim = 1

        X1 = Tensor(
            shape=[IntImm(M1), IntImm(K1)], dtype="float16", name="x1", is_input=True
        )
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        B1 = Tensor(shape=[N1], dtype="float16", name="b1", is_input=True)
        X2 = Tensor(
            shape=[IntImm(M2), IntImm(K2)], dtype="float16", name="x2", is_input=True
        )
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        B2 = Tensor(shape=[N2], dtype="float16", name="b2", is_input=True)

        Input = Tensor(
            shape=[BS, Input_M, Input_N], dtype="float16", name="input", is_input=True
        )

        group_gemm_op = ops.group_gemm_rcr_bias()
        Y1_orig, Y2_orig = group_gemm_op(operand_groups=[[X1, W1, B1], [X2, W2, B2]])
        Y1 = ops.reshape()(Y1_orig, [BS, -1, Input_N])
        Y2 = ops.unsqueeze(dim)(Y2_orig)
        Y1._attrs["name"] = "y1"
        Y2._attrs["name"] = "y2"
        concat_op = ops.concatenate()
        Y = concat_op([Y1, Y2, Input], dim=dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            [Y], target, "./tmp", "strided_reshape_cat_bias", dll_name=dll_name
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 2)
        np.testing.assert_equal(Y_src_ops, {group_gemm_op, concat_op})
        np.testing.assert_equal(concat_op._attrs["input_masks"], [False, False, True])
        expected_inputs_group_gemm_op = [X1, W1, B1, X2, W2, B2]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X1_pt = torch.randn(M1, K1).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        B1_pt = torch.randn(N1).cuda().half()
        X2_pt = torch.randn(M2, K2).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        B2_pt = torch.randn(N2).cuda().half()
        Input_pt = torch.randn(BS, Input_M, Input_N).cuda().half()
        Y1_orig_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_orig_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y1_pt = torch.reshape(Y1_orig_pt, [BS, -1, Input_N])
        Y2_pt = torch.unsqueeze(Y2_orig_pt, dim)
        Y_pt = torch.cat([Y1_pt, Y2_pt, Input_pt], dim=dim)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "b1": B1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "b2": B2_pt,
            "input": Input_pt,
        }
        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))
        self.test_count += 1


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
