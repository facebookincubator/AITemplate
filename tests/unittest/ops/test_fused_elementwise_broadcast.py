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
"""
Unittests for fused_elementwise broadcast.
"""
import itertools
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import graph_utils, shape_utils


class FusedElementwiseBroadcastTestCase(unittest.TestCase):
    def _test_different_dim(
        self,
        batch_sizes,
        ms,
        ks,
        test_name,
        expected_read_t,
        expected_op_t,
        expected_data_t,
        dtype="float16",
    ):
        """
        Tests tanh(A(B, M, K) + B(M, K)).
        """

        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes)
        m_dim = shape_utils.gen_int_var_min_max(ms)
        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[batch_dim, m_dim, k_dim],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[m_dim, k_dim],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X4 = ops.elementwise(FuncEnum.TANH)(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True
        self.assertEqual(X4._attrs["shape"], [batch_dim, m_dim, k_dim])

        target = detect_target()
        module = compile_model(X4, target, "./tmp", test_name)

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, k in itertools.product(batch_sizes, ms, ks):
            x1_pt = get_random_torch_tensor([batch_size, m, k], dtype=dtype)
            x2_pt = get_random_torch_tensor([m, k], dtype=dtype)
            x4_pt = torch.tanh(x1_pt + x2_pt)
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x4 = torch.empty_like(x4_pt)
            module.run_with_tensors(inputs, [x4])
            self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    def test_different_dim_fp16(self):
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp16_static_shapes",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_different_dim(
            batch_sizes=[23, 56, 1024],
            ms=[256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp16_dynamic_bs",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[34, 67, 256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp16_dynamic_ms",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[34, 87, 128],
            test_name="fused_elementwise_different_dim_fp16_dynamic_ks",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_different_dim(
            batch_sizes=[23, 1024],
            ms=[13, 256],
            ks=[34, 128],
            test_name="fused_elementwise_different_dim_fp16_dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_different_dim_fp32(self):
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp32_static_shapes",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_different_dim(
            batch_sizes=[23, 56, 1024],
            ms=[256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp32_dynamic_bs",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[34, 67, 256],
            ks=[128],
            test_name="fused_elementwise_different_dim_fp32_dynamic_ms",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[34, 87, 128],
            test_name="fused_elementwise_different_dim_fp32_dynamic_ks",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_different_dim(
            batch_sizes=[23, 1024],
            ms=[13, 256],
            ks=[34, 128],
            test_name="fused_elementwise_different_dim_fp32_dynamic_all",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )

    def _test_1_shape(
        self,
        batch_sizes,
        ms,
        ns,
        ks,
        test_name,
        expected_read_t,
        expected_op_t,
        expected_data_t,
        dtype="float16",
    ):
        """
        Tests tanh(A(B, 1, 1, M, K, 1) + B(N, N, 1, K, M)).
        """

        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes)
        m_dim = shape_utils.gen_int_var_min_max(ms)
        n_dim = shape_utils.gen_int_var_min_max(ns)
        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[batch_dim, IntImm(1), IntImm(1), m_dim, k_dim, IntImm(1)],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[n_dim, n_dim, IntImm(1), k_dim, m_dim],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X4 = ops.elementwise(FuncEnum.TANH)(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True
        self.assertEqual(
            X4._attrs["shape"], [batch_dim, n_dim, n_dim, m_dim, k_dim, m_dim]
        )

        target = detect_target()
        module = compile_model(X4, target, "./tmp", test_name)

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, n, k in itertools.product(batch_sizes, ms, ns, ks):
            x1_pt = get_random_torch_tensor([batch_size, 1, 1, m, k, 1], dtype=dtype)
            x2_pt = get_random_torch_tensor([n, n, 1, k, m], dtype=dtype)
            x4_pt = torch.tanh(x1_pt + x2_pt)
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x4 = torch.empty_like(x4_pt)
            module.run_with_tensors(inputs, [x4])
            self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    def test_1_shape_fp16(self):
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp16_static_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_1_shape(
            batch_sizes=[23, 56, 1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp16_dynamic_bs",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp16_dynamic_ms",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[1, 3, 4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp16_dynamic_ns",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="fused_elementwise_test_1_fp16_dynamic_ks",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_1_shape(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="fused_elementwise_test_1_fp16_dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_1_shape_fp32(self):
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp32_static_shapes",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_1_shape(
            batch_sizes=[23, 56, 1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp32_dynamic_bs",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp32_dynamic_ms",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[1, 3, 4],
            ks=[16],
            test_name="fused_elementwise_test_1_fp32_dynamic_ns",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="fused_elementwise_test_1_fp32_dynamic_ks",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_1_shape(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="fused_elementwise_test_1_fp32_dynamic_all",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )

    def _test_chained_broadcasts(
        self,
        batch_sizes,
        ms,
        ns,
        ks,
        test_name,
        expected_read_t,
        expected_op_t,
        expected_data_t,
        dtype="float16",
    ):
        """
        Tests A(B, 1, 1, M) + B(1, N, 1, M) + C(1, N, K, M).
        """

        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes)
        m_dim = shape_utils.gen_int_var_min_max(ms)
        n_dim = shape_utils.gen_int_var_min_max(ns)
        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[batch_dim, IntImm(1), IntImm(1), m_dim],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(1), n_dim, IntImm(1), m_dim],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X3 = Tensor(
            shape=[IntImm(1), n_dim, k_dim, m_dim],
            dtype=dtype,
            name="input2",
            is_input=True,
        )

        X4 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X5 = ops.elementwise(FuncEnum.ADD)(X3, X4)
        X5._attrs["name"] = "output0"
        X5._attrs["is_output"] = True
        self.assertEqual(X5._attrs["shape"], [batch_dim, n_dim, k_dim, m_dim])

        target = detect_target()
        module = compile_model(X5, target, "./tmp", test_name)

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, n, k in itertools.product(batch_sizes, ms, ns, ks):
            x1_pt = get_random_torch_tensor([batch_size, 1, 1, m], dtype=dtype)
            x2_pt = get_random_torch_tensor([1, n, 1, m], dtype=dtype)
            x3_pt = get_random_torch_tensor([1, n, k, m], dtype=dtype)
            x5_pt = x3_pt + x1_pt + x2_pt
            inputs = {"input0": x1_pt, "input1": x2_pt, "input2": x3_pt}
            x5 = torch.empty_like(x5_pt)
            module.run_with_tensors(inputs, [x5])
            self.assertTrue(torch.allclose(x5, x5_pt, atol=1e-2, rtol=1e-2))

    def test_chained_shapes_fp16(self):
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp16_static_shapes",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_chained_broadcasts(
            batch_sizes=[23, 56, 1024],
            ms=[2],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp16_dynamic_bs",
            expected_read_t="uint",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp16_dynamic_ms",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[4],
            ns=[1, 3, 4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp16_dynamic_ns",
            expected_read_t="uint2",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="fused_elementwise_chained_broadcasts_fp16_dynamic_ks",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_chained_broadcasts(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="fused_elementwise_chained_broadcasts_fp16_dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_chained_shapes_fp32(self):
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp32_static_shapes",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_chained_broadcasts(
            batch_sizes=[23, 56, 1024],
            ms=[2],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp32_dynamic_bs",
            expected_read_t="uint2",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp32_dynamic_ms",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[4],
            ns=[1, 3, 4],
            ks=[16],
            test_name="fused_elementwise_chained_broadcasts_fp32_dynamic_ns",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="fused_elementwise_chained_broadcasts_fp32_dynamic_ks",
            expected_read_t="uint4",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_chained_broadcasts(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="fused_elementwise_chained_broadcasts_fp32_dynamic_all",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )

    def _test_consecutive_1s_broadcast(
        self,
        ks,
        test_name,
        expected_read_t,
        expected_op_t,
        expected_data_t,
        dtype="float16",
    ):
        """
        Tests A(1, 1, K, 1, 1, K) / B(1, 1, 1, 1, 1, 1).
        """

        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[IntImm(1), IntImm(1), k_dim, IntImm(1), IntImm(1), k_dim],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(1), IntImm(1), IntImm(1), IntImm(1), IntImm(1), IntImm(1)],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.DIV)(X1, X2)
        X3._attrs["name"] = "output0"
        X3._attrs["is_output"] = True
        self.assertEqual(X3._attrs["shape"], X1._attrs["shape"])

        target = detect_target()
        module = compile_model(X3, target, "./tmp", test_name)

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for k in ks:
            x1_pt = get_random_torch_tensor([1, 1, k, 1, 1, k], dtype=dtype)
            x2_pt = get_random_torch_tensor([1, 1, 1, 1, 1, 1], dtype=dtype)
            x3_pt = x1_pt / x2_pt
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x3 = torch.empty_like(x3_pt)
            module.run_with_tensors(inputs, [x3])
            self.assertTrue(torch.allclose(x3, x3_pt, atol=1e-2, rtol=1e-2))

    def test_consecutive_1s_broadcast_fp16(self):
        self._test_consecutive_1s_broadcast(
            ks=[32],
            test_name="fused_elementwise_consecutive_1s_broadcast_fp16_static_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )
        self._test_consecutive_1s_broadcast(
            ks=[1, 5, 7, 32],
            test_name="fused_elementwise_consecutive_1s_broadcast_fp16_dynamic_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_consecutive_1s_broadcast_fp32(self):
        self._test_consecutive_1s_broadcast(
            ks=[32],
            test_name="fused_elementwise_consecutive_1s_broadcast_fp32_static_shapes",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )
        self._test_consecutive_1s_broadcast(
            ks=[1, 5, 7, 32],
            test_name="fused_elementwise_consecutive_1s_broadcast_fp32_dynamic_shapes",
            expected_read_t="float",
            expected_op_t="float",
            expected_data_t="float",
            dtype="float32",
        )


if __name__ == "__main__":
    unittest.main()
