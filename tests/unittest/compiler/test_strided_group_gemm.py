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
import os
import unittest

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.stable_set import StableSet
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


_LOGGER = logging.getLogger(__name__)


class StridedGroupGemmTestCase(unittest.TestCase):
    def _test_strided_group_gemm(
        self, M, N1, K1, N2, K2, N3, test_name, dtype="float16"
    ):
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return

        M1 = M
        M2 = M
        M3 = M

        dim = 1

        X1 = Tensor(
            shape=[IntImm(M1), IntImm(K1)], dtype=dtype, name="x1", is_input=True
        )
        W1 = Tensor(shape=[N1, K1], dtype=dtype, name="w1", is_input=True)
        X2 = Tensor(
            shape=[IntImm(M2), IntImm(K2)], dtype=dtype, name="x2", is_input=True
        )
        W2 = Tensor(shape=[N2, K2], dtype=dtype, name="w2", is_input=True)

        X3 = Tensor(shape=[M3, N3], dtype=dtype, name="x3", is_input=True)

        group_gemm_op = ops.group_gemm_rcr()
        Y1, Y2 = group_gemm_op(operand_groups=[[X1, W1], [X2, W2]])
        Y1._attrs["name"] = "y1"
        Y2._attrs["name"] = "y2"
        concat_op = ops.concatenate()
        Y = concat_op([Y1, Y2, X3], dim=dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        module = compile_model([Y], target, "./tmp", test_name)
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 2)
        np.testing.assert_equal(Y_src_ops, StableSet({group_gemm_op, concat_op}))
        expected_inputs_group_gemm_op = [X1, W1, X2, W2]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X1_pt = get_random_torch_tensor([M1, K1], dtype)
        W1_pt = get_random_torch_tensor([N1, K1], dtype)
        X2_pt = get_random_torch_tensor([M2, K2], dtype)
        W2_pt = get_random_torch_tensor([N2, K2], dtype)
        X3_pt = get_random_torch_tensor([M3, N3], dtype)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)
        Y_pt = torch.cat([Y1_pt, Y2_pt, X3_pt], dim=dim)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "x3": X3_pt,
        }
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_strided_group_gemm(self):
        self._test_strided_group_gemm(
            M=128,
            N1=32,
            K1=32,
            N2=64,
            K2=16,
            N3=8,
            test_name="strided_group_gemm_rcr_cat1",
        )
        self._test_strided_group_gemm(
            M=8, N1=32, K1=32, N2=4, K2=4, N3=3, test_name="strided_group_gemm_rcr_cat2"
        )

    def _test_strided_group_gemm_bias(
        self, M, N1, K1, N2, K2, N3, test_name, input_first, dtype="float16"
    ):
        # input_first determines if we place input tensor (X3) to be the first
        # concatenated tensor or not
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return
        M1 = M
        M2 = M
        M3 = M

        dim = 1

        X1 = Tensor(
            shape=[IntImm(M1), IntImm(K1)], dtype=dtype, name="x1", is_input=True
        )
        W1 = Tensor(shape=[N1, K1], dtype=dtype, name="w1", is_input=True)
        B1 = Tensor(shape=[N1], dtype=dtype, name="b1", is_input=True)
        X2 = Tensor(
            shape=[IntImm(M2), IntImm(K2)], dtype=dtype, name="x2", is_input=True
        )
        W2 = Tensor(shape=[N2, K2], dtype=dtype, name="w2", is_input=True)
        B2 = Tensor(shape=[N2], dtype=dtype, name="b2", is_input=True)

        X3 = Tensor(shape=[M3, N3], dtype=dtype, name="x3", is_input=True)

        group_gemm_op = ops.group_gemm_rcr_bias()
        Y1, Y2 = group_gemm_op(operand_groups=[[X1, W1, B1], [X2, W2, B2]])
        Y1._attrs["name"] = "y1"
        Y2._attrs["name"] = "y2"
        concat_op = ops.concatenate()
        if input_first:
            Y = concat_op([X3, Y1, Y2], dim=dim)
        else:
            Y = concat_op([Y1, Y2, X3], dim=dim)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True
        module = compile_model(
            [Y],
            target,
            "./tmp",
            test_name,
        )
        Y_src_ops = Y._attrs["src_ops"]
        np.testing.assert_equal(len(Y_src_ops), 2)
        np.testing.assert_equal(Y_src_ops, StableSet({group_gemm_op, concat_op}))
        expected_inputs_group_gemm_op = [X1, W1, B1, X2, W2, B2]
        np.testing.assert_equal(
            expected_inputs_group_gemm_op, group_gemm_op._attrs["inputs"]
        )

        X1_pt = get_random_torch_tensor([M1, K1], dtype)
        W1_pt = get_random_torch_tensor([N1, K1], dtype)
        B1_pt = get_random_torch_tensor([N1], dtype)
        X2_pt = get_random_torch_tensor([M2, K2], dtype)
        W2_pt = get_random_torch_tensor([N2, K2], dtype)
        B2_pt = get_random_torch_tensor([N2], dtype)
        X3_pt = get_random_torch_tensor([M3, N3], dtype)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)
        if input_first:
            Y_pt = torch.cat([X3_pt, Y1_pt, Y2_pt], dim=dim)
        else:
            Y_pt = torch.cat([Y1_pt, Y2_pt, X3_pt], dim=dim)

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        _LOGGER.info("AITemplate y_shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "b1": B1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
            "b2": B2_pt,
            "x3": X3_pt,
        }
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_strided_group_gemm_bias(self):
        self._test_strided_group_gemm_bias(
            M=128,
            N1=32,
            K1=32,
            N2=64,
            K2=16,
            N3=8,
            test_name="strided_group_gemm_rcr_bias_cat1",
            input_first=False,
        )
        self._test_strided_group_gemm_bias(
            M=8,
            N1=32,
            K1=32,
            N2=4,
            K2=4,
            N3=3,
            test_name="strided_group_gemm_rcr_bias_cat2",
            input_first=False,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_strided_group_gemm_float(self):
        self._test_strided_group_gemm(
            M=8,
            N1=32,
            K1=32,
            N2=4,
            K2=4,
            N3=3,
            test_name="strided_group_gemm_rcr_cat_float2",
            dtype="float",
        )
        self._test_strided_group_gemm_bias(
            M=128,
            N1=32,
            K1=32,
            N2=64,
            K2=16,
            N3=8,
            test_name="strided_group_gemm_rcr_bias_cat_float1",
            input_first=False,
            dtype="float",
        )

    # test if we update epilogue alignment values correctly
    def _test_strided_group_gemm_epilogue_alignment(self, dtype="float16"):
        # Note that we have to force profiling in ci. Otherwise, we would not
        # be able to fetch cached config.
        target = detect_target()
        old_force_ci = os.environ.get("FORCE_PROFILE", None)
        if target.in_ci_env():
            os.environ["FORCE_PROFILE"] = "1"

        # a smaller epilogue alignment value 2
        self._test_strided_group_gemm_bias(
            M=18,
            N1=24,
            K1=32,
            N2=62,
            K2=16,
            N3=2,
            test_name=f"strided_group_gemm_rcr_epilogue_alignment_{dtype}_1",
            input_first=True,
            dtype=dtype,
        )
        # a bigger epilogue alignment value 4
        self._test_strided_group_gemm_bias(
            M=18,
            N1=24,
            K1=32,
            N2=62,
            K2=16,
            N3=4,
            test_name=f"strided_group_gemm_rcr_epilogue_alignment_{dtype}_2",
            input_first=True,
            dtype=dtype,
        )

        # restore old env
        if target.in_ci_env():
            if old_force_ci is None:
                del os.environ["FORCE_PROFILE"]
            else:
                os.environ["FORCE_PROFILE"] = old_force_ci

    def test_strided_group_gemm_epilogue_alignment(self):
        self._test_strided_group_gemm_epilogue_alignment()

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_strided_group_gemm_epilogue_alignment_float(self):
        self._test_strided_group_gemm_epilogue_alignment(dtype="float")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
