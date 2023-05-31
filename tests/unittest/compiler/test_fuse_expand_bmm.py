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
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class FuseExpandBmmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(FuseExpandBmmTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _compile_and_check(
        self, Y, test_name, expected_num_ops, expected_op, no_expand=True
    ):
        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        self.test_count += 1
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)
        if expected_num_ops == 1:
            self.assertEqual(sorted_ops[0]._attrs["op"], expected_op)
        elif no_expand:
            self.assertTrue(
                all(lambda op: op._attrs["op"] != "expand" for op in sorted_ops)
            )
        return module

    def _test_non_fusible_expand_bmm_1(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, M, K])
        # x1 = tensor([B, K, N])
        # Y0 = expand(x0, shape_1[B, M, K])
        # Y1 = bmm_rrr(Y_0, x1)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="x1", is_input=True)

        Y0 = ops.expand()(X0, [batch_dim, -1, -1])
        Y0._attrs["name"] = "output0"
        Y0._attrs["is_output"] = True

        Y1 = ops.bmm_rrr()(Y0, X1)
        Y1._attrs["name"] = "output1"
        Y1._attrs["is_output"] = True
        module = self._compile_and_check(
            [Y0, Y1], test_name, expected_num_ops, "bmm_rrr", no_expand=False
        )

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, M, K], dtype)
            x1_pt = get_random_torch_tensor([batch, K, N], dtype)
            y0_pt = x0_pt.expand(batch, -1, -1)
            y1_pt = torch.matmul(y0_pt, x1_pt)

            y0 = get_torch_empty_tensor(y0_pt.size(), dtype)
            y1 = get_torch_empty_tensor(y1_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y0, y1])
            torch.testing.assert_close(y0_pt, y0, atol=0.1, rtol=0.1)
            torch.testing.assert_close(y1_pt, y1, atol=0.1, rtol=0.1)

    def test_non_fusible_expand_bmm_1(self):
        self._test_non_fusible_expand_bmm_1(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_non_fusible_expand_bmm_1",
        )

    def _test_non_fusible_expand_bmm_2(
        self,
        B,
        M,
        N,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, M, N])
        # x1 = tensor([B, N, N])
        # expand_0 = expand(x0, shape_1[B, M, N])
        # bmm_rrr_1 = bmm_rrr(expand_0, x1)
        # Y = add(expand_0, bmm_rrr_1)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, M, N], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[batch_dim, N, N], dtype=dtype, name="x1", is_input=True)

        expand_0 = ops.expand()(X0, [batch_dim, -1, -1])
        bmm_rrr_1 = ops.bmm_rrr()(expand_0, X1)
        Y = ops.elementwise(FuncEnum.ADD)(expand_0, bmm_rrr_1)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(
            Y, test_name, expected_num_ops, "bmm_rrr", no_expand=False
        )

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, M, N], dtype)
            x1_pt = get_random_torch_tensor([batch, N, N], dtype)
            expand_0_pt = x0_pt.expand(batch, -1, -1)
            bmm_rrr_1_pt = torch.matmul(expand_0_pt, x1_pt)
            y_pt = expand_0_pt + bmm_rrr_1_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_non_fusible_expand_bmm_2(self):
        self._test_non_fusible_expand_bmm_1(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_non_fusible_expand_bmm_1",
        )

    def _test_fuse_expand_bmm_rrr_a(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, M, K])
        # x1 = tensor([B, K, N])
        # expand_0 = expand(x0, shape_1[B, M, K])
        # Y = bmm_rrr(expand_0, x1)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="x1", is_input=True)

        expand_0 = ops.expand()(X0, [batch_dim, -1, -1])
        Y = ops.bmm_rrr()(expand_0, X1)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_rrr")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, M, K], dtype)
            x1_pt = get_random_torch_tensor([batch, K, N], dtype)
            expand_0_pt = x0_pt.expand(batch, -1, -1)
            y_pt = torch.matmul(expand_0_pt, x1_pt)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_rrr_a(self):
        self._test_fuse_expand_bmm_rrr_a(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=2,  # one extra permute
            test_name="test_fuse_expand_bmm_rrr_a",
        )
        self._test_fuse_expand_bmm_rrr_a(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=1,
            test_name="test_fuse_expand_bmm_rrr_a",
        )

    def _test_fuse_expand_bmm_rrc_add_b(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([B, M, K])
        # x1 = tensor([1, K, N])
        # x2 = tensor([B, N, M])
        # expand_0 = expand(x1, shape_1[B, K, N])
        # Y = bmm_rrc_add(x0, expand_0, x2)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[batch_dim, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, K, N], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[batch_dim, N, M], dtype=dtype, name="x2", is_input=True)

        expand_0 = ops.expand()(X1, [batch_dim, -1, -1])
        Y = ops.bmm_rrc_add()(X0, expand_0, X2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_rrc_add")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([batch, M, K], dtype)
            x1_pt = get_random_torch_tensor([1, K, N], dtype)
            x2_pt = get_random_torch_tensor([batch, N, M], dtype)
            expand_0_pt = x1_pt.expand(batch, -1, -1)
            y_pt = torch.matmul(x0_pt, expand_0_pt)
            y_pt = y_pt.transpose(2, 1) + x2_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_rrc_add_b(self):
        self._test_fuse_expand_bmm_rrc_add_b(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=3,  # two extra concat
            test_name="test_fuse_expand_bmm_rrc_add_b",
        )
        self._test_fuse_expand_bmm_rrc_add_b(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=1,
            test_name="test_fuse_expand_bmm_rrc_add_b",
        )

    def _test_fuse_expand_bmm_crr_a(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, K, M])
        # x1 = tensor([1, K, M])
        # x2 = tensor([B, K, N])
        # add_0 = x0 + x1
        # expand_0 = expand(add_0, shape_1[B, K, M])
        # Y = bmm_rrr(expand_0, x2)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, K, M], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, K, M], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="x2", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_crr()(expand_1, X2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_crr")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, K, M], dtype)
            x1_pt = get_random_torch_tensor([1, K, M], dtype)
            x2_pt = get_random_torch_tensor([batch, K, N], dtype)
            add_0_pt = x0_pt + x1_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            expand_1_tran_pt = torch.transpose(expand_1_pt, 2, 1)
            y_pt = torch.matmul(expand_1_tran_pt, x2_pt)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_crr_a(self):
        self._test_fuse_expand_bmm_crr_a(
            B=10,
            M=5,
            N=12,
            K=11,
            expected_num_ops=4,  # extra concat and slice
            test_name="test_fuse_expand_bmm_crr_a",
        )
        self._test_fuse_expand_bmm_crr_a(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_crr_a",
        )

    def _test_fuse_expand_bmm_crc_add_b(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([B, M, K])
        # x1 = tensor([1, K, N])
        # x2 = tensor([1, K, N])
        # x3 = tensor([B, N, M])
        # add_0 = x1 + x2
        # expand_0 = expand(add_0, shape_1[B, K, N])
        # Y = bmm_rrc_add(x0, expand_0, x3)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[batch_dim, K, M], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, K, N], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[1, K, N], dtype=dtype, name="x2", is_input=True)
        X3 = Tensor(shape=[batch_dim, N, M], dtype=dtype, name="x3", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_crc_add()(X0, expand_1, X3)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_crc_add")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([batch, K, M], dtype)
            x1_pt = get_random_torch_tensor([1, K, N], dtype)
            x2_pt = get_random_torch_tensor([1, K, N], dtype)
            x3_pt = get_random_torch_tensor([batch, N, M], dtype)
            add_0_pt = x1_pt + x2_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            x0_tran_pt = torch.transpose(x0_pt, 2, 1)
            y_pt = torch.matmul(x0_tran_pt, expand_1_pt)
            y_pt = y_pt.transpose(2, 1) + x3_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_crc_add_b(self):
        self._test_fuse_expand_bmm_crc_add_b(
            B=10,
            M=5,
            N=12,
            K=6,
            expected_num_ops=5,  # two extra concat and one slice
            test_name="test_fuse_expand_bmm_crc_add_b",
        )
        self._test_fuse_expand_bmm_crc_add_b(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_crc_add_b",
        )

    def _test_fuse_expand_bmm_rcr_a(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, M, K])
        # x1 = tensor([1, M, K])
        # x2 = tensor([B, N, K])
        # add_0 = x0 + x1
        # expand_0 = expand(add_0, shape_1[B, M, K])
        # Y = bmm_rrr(expand_0, x2)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, M, K], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[batch_dim, N, K], dtype=dtype, name="x2", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_rcr()(expand_1, X2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_rcr")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, M, K], dtype)
            x1_pt = get_random_torch_tensor([1, M, K], dtype)
            x2_pt = get_random_torch_tensor([batch, N, K], dtype)
            add_0_pt = x0_pt + x1_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            x2_tran_pt = torch.transpose(x2_pt, 2, 1)
            y_pt = torch.matmul(expand_1_pt, x2_tran_pt)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_rcr_a(self):
        self._test_fuse_expand_bmm_rcr_a(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=4,
            test_name="test_fuse_expand_bmm_rcr_a",
        )
        self._test_fuse_expand_bmm_rcr_a(
            B=10,
            M=5,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_rcr_a",
        )

    def _test_fuse_expand_bmm_rcc_add_b(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([B, M, K])
        # x1 = tensor([1, N, K])
        # x2 = tensor([1, N, K])
        # x3 = tensor([B, N, M])
        # add_0 = x1 + x2
        # expand_0 = expand(add_0, shape_1[B, N, K])
        # Y = bmm_rrc_add(x0, expand_0, x3)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[batch_dim, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, N, K], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[1, N, K], dtype=dtype, name="x2", is_input=True)
        X3 = Tensor(shape=[batch_dim, N, M], dtype=dtype, name="x3", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_rcc_add()(X0, expand_1, X3)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_rcc_add")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([batch, M, K], dtype)
            x1_pt = get_random_torch_tensor([1, N, K], dtype)
            x2_pt = get_random_torch_tensor([1, N, K], dtype)
            x3_pt = get_random_torch_tensor([batch, N, M], dtype)
            add_0_pt = x1_pt + x2_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            expand_1_tran_pt = torch.transpose(expand_1_pt, 2, 1)
            y_pt = torch.matmul(x0_pt, expand_1_tran_pt)
            y_pt = y_pt.transpose(2, 1) + x3_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_rcc_add_b(self):
        self._test_fuse_expand_bmm_rcc_add_b(
            B=10,
            M=6,
            N=12,
            K=5,
            expected_num_ops=4,  # two extra concat
            test_name="test_fuse_expand_bmm_rcc_add_b",
        )
        self._test_fuse_expand_bmm_rcc_add_b(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_rcc_add_b",
        )

    def _test_fuse_expand_bmm_ccr_a(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, K, M])
        # x1 = tensor([1, K, M])
        # x2 = tensor([B, N, K])
        # add_0 = x0 + x1
        # expand_0 = expand(add_0, shape_1[B, K, M])
        # Y = bmm_rrr(expand_0, x2)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, K, M], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, K, M], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[batch_dim, N, K], dtype=dtype, name="x2", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X0, X1)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_ccr()(expand_1, X2)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_ccr")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, K, M], dtype)
            x1_pt = get_random_torch_tensor([1, K, M], dtype)
            x2_pt = get_random_torch_tensor([batch, N, K], dtype)
            add_0_pt = x0_pt + x1_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            expand_1_tran_pt = torch.transpose(expand_1_pt, 2, 1)
            x2_tran_pt = torch.transpose(x2_pt, 2, 1)
            y_pt = torch.matmul(expand_1_tran_pt, x2_tran_pt)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_ccr_a(self):
        self._test_fuse_expand_bmm_ccr_a(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=3,  # one extra permute
            test_name="test_fuse_expand_bmm_ccr_a",
        )
        self._test_fuse_expand_bmm_ccr_a(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_ccr_a",
        )

    def _test_fuse_expand_bmm_ccc_add_b(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([B, K, M])
        # x1 = tensor([1, N, K])
        # x2 = tensor([1, N, K])
        # x3 = tensor([B, N, M])
        # add_0 = x1 + x2
        # expand_0 = expand(add_0, shape_1[B, N, K])
        # Y = bmm_rrc_add(x0, expand_0, x3)
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[batch_dim, K, M], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, N, K], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[1, N, K], dtype=dtype, name="x2", is_input=True)
        X3 = Tensor(shape=[batch_dim, N, M], dtype=dtype, name="x3", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        expand_1 = ops.expand()(add_0, [batch_dim, -1, -1])
        Y = ops.bmm_ccc_add()(X0, expand_1, X3)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_ccc_add")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([batch, K, M], dtype)
            x1_pt = get_random_torch_tensor([1, N, K], dtype)
            x2_pt = get_random_torch_tensor([1, N, K], dtype)
            x3_pt = get_random_torch_tensor([batch, N, M], dtype)
            add_0_pt = x1_pt + x2_pt
            expand_1_pt = add_0_pt.expand(batch, -1, -1)
            expand_1_tran_pt = torch.transpose(expand_1_pt, 2, 1)
            x0_tran_pt = torch.transpose(x0_pt, 2, 1)
            y_pt = torch.matmul(x0_tran_pt, expand_1_tran_pt)
            y_pt = y_pt.transpose(2, 1) + x3_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_expand_bmm_ccc_add_b(self):
        self._test_fuse_expand_bmm_ccc_add_b(
            B=10,
            M=5,
            N=12,
            K=6,
            expected_num_ops=5,  # two extra concat and one slice
            test_name="test_fuse_expand_bmm_ccc_add_b",
        )
        self._test_fuse_expand_bmm_ccc_add_b(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=2,
            test_name="test_fuse_expand_bmm_ccc_add_b",
        )

    def _test_fuse_size_expand_bmm_rrr(
        self,
        B,
        M,
        N,
        K,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # make a graph like below:
        # x0 = tensor([1, M, K])
        # x1 = tensor([1, M, K])
        # x2 = tensor([1, M, K])
        # x3 = tensor([B, K, N])
        # x4 = tensor([B, K, N])
        # add_0 = x3 + x4
        # size_1, _, _ = size(add_0)
        # expand_2 = expand(x0, size_1)
        # expand_3 = expand(x1, size_1)
        # expand_4 = expand(x2, size_1)
        # bmm_5 = bmm_rrr(expand_2, add_0)
        # bmm_6 = bmm_rrr(expand_3, add_0)
        # bmm_7 = bmm_rrr(expand_4, add_0)
        # add_8 = bmm_5 + bmm_6
        # Y = bmm_7 + add_8
        batch_sizes = [1, B]
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        X0 = Tensor(shape=[1, M, K], dtype=dtype, name="x0", is_input=True)
        X1 = Tensor(shape=[1, M, K], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[1, M, K], dtype=dtype, name="x2", is_input=True)
        X3 = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="x3", is_input=True)
        X4 = Tensor(shape=[batch_dim, K, N], dtype=dtype, name="x4", is_input=True)

        add_0 = ops.elementwise(FuncEnum.ADD)(X3, X4)
        size_1, _, _ = ops.size()(add_0)
        expand_to_shape = [size_1, -1, -1]
        expand_2 = ops.expand()(X0, expand_to_shape)
        expand_3 = ops.expand()(X1, expand_to_shape)
        expand_4 = ops.expand()(X2, expand_to_shape)
        bmm_5 = ops.bmm_rrr()(expand_2, add_0)
        bmm_6 = ops.bmm_rrr()(expand_3, add_0)
        bmm_7 = ops.bmm_rrr()(expand_4, add_0)
        add_8 = ops.elementwise(FuncEnum.ADD)(bmm_5, bmm_6)
        Y = ops.elementwise(FuncEnum.ADD)(bmm_7, add_8)
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True
        module = self._compile_and_check(Y, test_name, expected_num_ops, "bmm_rrr")

        for batch in [1, B]:
            x0_pt = get_random_torch_tensor([1, M, K], dtype)
            x1_pt = get_random_torch_tensor([1, M, K], dtype)
            x2_pt = get_random_torch_tensor([1, M, K], dtype)
            x3_pt = get_random_torch_tensor([batch, K, N], dtype)
            x4_pt = get_random_torch_tensor([batch, K, N], dtype)
            add_0_pt = x3_pt + x4_pt
            size_1 = batch
            expand_2_pt = x0_pt.expand(size_1, -1, -1)
            expand_3_pt = x1_pt.expand(size_1, -1, -1)
            expand_4_pt = x2_pt.expand(size_1, -1, -1)
            bmm_5_pt = torch.matmul(expand_2_pt, add_0_pt)
            bmm_6_pt = torch.matmul(expand_3_pt, add_0_pt)
            bmm_7_pt = torch.matmul(expand_4_pt, add_0_pt)
            add_8_pt = bmm_5_pt + bmm_6_pt
            y_pt = bmm_7_pt + add_8_pt

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            inputs = {"x0": x0_pt, "x1": x1_pt, "x2": x2_pt, "x3": x3_pt, "x4": x4_pt}
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y_pt, y, atol=0.1, rtol=0.1)

    def test_fuse_size_expand_bmm_rrr(self):
        self._test_fuse_size_expand_bmm_rrr(
            B=10,
            M=4,
            N=12,
            K=11,
            expected_num_ops=7,
            test_name="test_fuse_size_expand_bmm_rrr",
        )
        self._test_fuse_size_expand_bmm_rrr(
            B=10,
            M=4,
            N=12,
            K=6,
            expected_num_ops=4,
            test_name="test_fuse_size_expand_bmm_rrr",
        )


if __name__ == "__main__":
    unittest.main()
