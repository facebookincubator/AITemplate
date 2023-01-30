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
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


_LOGGER = logging.getLogger(__name__)


class StridedSplitGroupGemmTestCase(unittest.TestCase):
    def _test_split_group_gemm(self, dtype="float16"):
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
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
            dtype=dtype,
            name="x",
            is_input=True,
        )
        W1 = Tensor(shape=[N, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype=dtype, name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype=dtype, name="w3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr()
        Y = group_gemm_op(
            operand_groups=[[X1, W1], [X2, W2], [X3, W3]], output_stride_dim=dim
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_cat.so"
        module = compile_model(
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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W1_pt = get_random_torch_tensor([N, K1], dtype)
        W2_pt = get_random_torch_tensor([N, K2], dtype)
        W3_pt = get_random_torch_tensor([N, K3], dtype)
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        inputs = [
            X_pt,
            W1_pt,
            W2_pt,
            W3_pt,
        ]
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_split_group_gemm_bias(self, dtype="float16"):
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(shape=[IntImm(M), IntImm(K)], dtype=dtype, name="x", is_input=True)
        W1 = Tensor(shape=[N, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype=dtype, name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype=dtype, name="w3", is_input=True)
        B1 = Tensor(shape=[N], dtype=dtype, name="b1", is_input=True)
        B2 = Tensor(shape=[N], dtype=dtype, name="b2", is_input=True)
        B3 = Tensor(shape=[N], dtype=dtype, name="b3", is_input=True)

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
        module = compile_model(
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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W1_pt = get_random_torch_tensor([N, K1], dtype)
        W2_pt = get_random_torch_tensor([N, K2], dtype)
        W3_pt = get_random_torch_tensor([N, K3], dtype)
        B1_pt = get_random_torch_tensor([N], dtype)
        B2_pt = get_random_torch_tensor([N], dtype)
        B3_pt = get_random_torch_tensor([N], dtype)
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt, bias=B3_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0] * 7
        inputs[input_name_to_index["x"]] = X_pt
        inputs[input_name_to_index["w1"]] = W1_pt
        inputs[input_name_to_index["w2"]] = W2_pt
        inputs[input_name_to_index["w3"]] = W3_pt
        inputs[input_name_to_index["b1"]] = B1_pt
        inputs[input_name_to_index["b2"]] = B2_pt
        inputs[input_name_to_index["b3"]] = B3_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_split_group_gemm_reorder(self, dtype="float16"):
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(shape=[IntImm(M), IntImm(K)], dtype=dtype, name="x", is_input=True)
        W1 = Tensor(shape=[N, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype=dtype, name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype=dtype, name="w3", is_input=True)

        split_op = ops.split()
        X1, X2, X3 = split_op(X, [K1, K2, K3], dim)
        group_gemm_op = ops.group_gemm_rcr()
        Y = group_gemm_op(
            operand_groups=[[X2, W2], [X1, W1], [X3, W3]], output_stride_dim=dim
        )
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        dll_name = "test_rcr_cat_reorder.so"
        module = compile_model(
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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W1_pt = get_random_torch_tensor([N, K1], dtype)
        W2_pt = get_random_torch_tensor([N, K2], dtype)
        W3_pt = get_random_torch_tensor([N, K3], dtype)
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt)
        Y_pt = torch.cat([Y2_pt, Y1_pt, Y3_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        inputs = [0 for i in range(4)]
        name_to_idx = module.get_input_name_to_index_map()
        inputs[name_to_idx["x"]] = X_pt
        inputs[name_to_idx["w1"]] = W1_pt
        inputs[name_to_idx["w2"]] = W2_pt
        inputs[name_to_idx["w3"]] = W3_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def _test_split_group_gemm_bias_reorder(self, dtype="float16"):
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return

        K1 = 32
        K2 = 16
        K3 = 64

        M = 128
        N = 32
        K = K1 + K2 + K3

        dim = 1

        X = Tensor(shape=[IntImm(M), IntImm(K)], dtype=dtype, name="x", is_input=True)
        W1 = Tensor(shape=[N, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N, K2], dtype=dtype, name="w2", is_input=True)
        W3 = Tensor(shape=[N, K3], dtype=dtype, name="w3", is_input=True)
        B1 = Tensor(shape=[N], dtype=dtype, name="b1", is_input=True)
        B2 = Tensor(shape=[N], dtype=dtype, name="b2", is_input=True)
        B3 = Tensor(shape=[N], dtype=dtype, name="b3", is_input=True)

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
        module = compile_model(
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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W1_pt = get_random_torch_tensor([N, K1], dtype)
        W2_pt = get_random_torch_tensor([N, K2], dtype)
        W3_pt = get_random_torch_tensor([N, K3], dtype)
        B1_pt = get_random_torch_tensor([N], dtype)
        B2_pt = get_random_torch_tensor([N], dtype)
        B3_pt = get_random_torch_tensor([N], dtype)
        X1_pt, X2_pt, X3_pt = torch.split(X_pt, [K1, K2, K3], dim)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        Y3_pt = torch.nn.functional.linear(X3_pt, W3_pt, bias=B3_pt)
        Y_pt = torch.cat([Y2_pt, Y3_pt, Y1_pt], dim=dim)
        Y_np = Y_pt.cpu().numpy()

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_np.shape)

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0] * 7
        inputs[input_name_to_index["x"]] = X_pt
        inputs[input_name_to_index["w1"]] = W1_pt
        inputs[input_name_to_index["w2"]] = W2_pt
        inputs[input_name_to_index["w3"]] = W3_pt
        inputs[input_name_to_index["b1"]] = B1_pt
        inputs[input_name_to_index["b2"]] = B2_pt
        inputs[input_name_to_index["b3"]] = B3_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_split_group_gemm_float16(self):
        self._test_split_group_gemm()
        self._test_split_group_gemm_bias()
        self._test_split_group_gemm_reorder()
        self._test_split_group_gemm_bias_reorder()

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_split_group_gemm_float(self):
        self._test_split_group_gemm(dtype="float")
        self._test_split_group_gemm_bias(dtype="float")
        self._test_split_group_gemm_reorder(dtype="float")
        self._test_split_group_gemm_bias_reorder(dtype="float")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
