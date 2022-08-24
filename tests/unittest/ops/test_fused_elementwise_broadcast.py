# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for fused_elementwise broadcast.
"""
import itertools
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
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
    ):
        """
        Tests tanh(A(B, M, K) + B(M, K)).
        """

        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes)
        m_dim = shape_utils.gen_int_var_min_max(ms)
        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[batch_dim, m_dim, k_dim],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[m_dim, k_dim],
            dtype="float16",
            name="input1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X4 = ops.elementwise(FuncEnum.TANH)(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True
        self.assertEqual(X4._attrs["shape"], [batch_dim, m_dim, k_dim])

        target = detect_target()
        module = gen_execution_module(
            X4,
            target,
            "./tmp",
            "fused_elementwise_different_dims_{}".format(test_name),
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, k in itertools.product(batch_sizes, ms, ks):
            x1_pt = torch.randn(batch_size, m, k).cuda().half()
            x2_pt = torch.randn(m, k).cuda().half()
            x4_pt = torch.tanh(x1_pt + x2_pt)
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x4 = torch.empty([batch_size, m, k]).cuda().half()
            module.RunWithTensors(inputs, [x4])
            self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    def test_different_dim(self):
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[128],
            test_name="static_shapes",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_different_dim(
            batch_sizes=[23, 56, 1024],
            ms=[256],
            ks=[128],
            test_name="dynamic_bs",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[34, 67, 256],
            ks=[128],
            test_name="dynamic_ms",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_different_dim(
            batch_sizes=[1024],
            ms=[256],
            ks=[34, 87, 128],
            test_name="dynamic_ks",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_different_dim(
            batch_sizes=[23, 1024],
            ms=[13, 256],
            ks=[34, 128],
            test_name="dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
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
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[n_dim, n_dim, IntImm(1), k_dim, m_dim],
            dtype="float16",
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
        module = gen_execution_module(
            X4,
            target,
            "./tmp",
            "fused_elementwise_1_shape_{}".format(test_name),
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, n, k in itertools.product(batch_sizes, ms, ns, ks):
            x1_pt = torch.randn(batch_size, 1, 1, m, k, 1).cuda().half()
            x2_pt = torch.randn(n, n, 1, k, m).cuda().half()
            x4_pt = torch.tanh(x1_pt + x2_pt)
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x4 = torch.empty([batch_size, n, n, m, k, m]).cuda().half()
            module.RunWithTensors(inputs, [x4])
            self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    def test_1_shape(self):
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="static_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_1_shape(
            batch_sizes=[23, 56, 1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="dynamic_bs",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="dynamic_ms",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[1, 3, 4],
            ks=[16],
            test_name="dynamic_ns",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_1_shape(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="dynamic_ks",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_1_shape(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
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
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(1), n_dim, IntImm(1), m_dim],
            dtype="float16",
            name="input1",
            is_input=True,
        )
        X3 = Tensor(
            shape=[IntImm(1), n_dim, k_dim, m_dim],
            dtype="float16",
            name="input2",
            is_input=True,
        )

        X4 = ops.elementwise(FuncEnum.ADD)(X1, X2)
        X5 = ops.elementwise(FuncEnum.ADD)(X3, X4)
        X5._attrs["name"] = "output0"
        X5._attrs["is_output"] = True
        self.assertEqual(X5._attrs["shape"], [batch_dim, n_dim, k_dim, m_dim])

        target = detect_target()
        module = gen_execution_module(
            X5,
            target,
            "./tmp",
            "fused_elementwise_chained_broadcasts_{}".format(test_name),
        )

        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for batch_size, m, n, k in itertools.product(batch_sizes, ms, ns, ks):
            x1_pt = torch.randn(batch_size, 1, 1, m).cuda().half()
            x2_pt = torch.randn(1, n, 1, m).cuda().half()
            x3_pt = torch.randn(1, n, k, m).cuda().half()
            x5_pt = x3_pt + x1_pt + x2_pt
            inputs = {"input0": x1_pt, "input1": x2_pt, "input2": x3_pt}
            x5 = torch.empty([batch_size, n, k, m]).cuda().half()
            module.RunWithTensors(inputs, [x5])
            self.assertTrue(torch.allclose(x5, x5_pt, atol=1e-2, rtol=1e-2))

    def test_chained_shapes(self):
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[16],
            test_name="static_shapes",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_chained_broadcasts(
            batch_sizes=[23, 56, 1024],
            ms=[2],
            ns=[4],
            ks=[16],
            test_name="dynamic_bs",
            expected_read_t="uint",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[1, 3, 8],
            ns=[4],
            ks=[16],
            test_name="dynamic_ms",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[4],
            ns=[1, 3, 4],
            ks=[16],
            test_name="dynamic_ns",
            expected_read_t="uint2",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_chained_broadcasts(
            batch_sizes=[1024],
            ms=[8],
            ns=[4],
            ks=[1, 4, 7, 16],
            test_name="dynamic_ks",
            expected_read_t="uint4",
            expected_op_t="half2",
            expected_data_t="half",
        )
        self._test_chained_broadcasts(
            batch_sizes=[25, 1024],
            ms=[7, 8],
            ns=[3, 4],
            ks=[1, 16],
            test_name="dynamic_all",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )

    def _test_consecutive_1s_broadcast(
        self,
        ks,
        test_name,
        expected_read_t,
        expected_op_t,
        expected_data_t,
    ):
        """
        Tests A(1, 1, K, 1, 1, K) / B(1, 1, 1, 1, 1, 1).
        """

        k_dim = shape_utils.gen_int_var_min_max(ks)

        X1 = Tensor(
            shape=[IntImm(1), IntImm(1), k_dim, IntImm(1), IntImm(1), k_dim],
            dtype="float16",
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(1), IntImm(1), IntImm(1), IntImm(1), IntImm(1), IntImm(1)],
            dtype="float16",
            name="input1",
            is_input=True,
        )
        X3 = ops.elementwise(FuncEnum.DIV)(X1, X2)
        X3._attrs["name"] = "output0"
        X3._attrs["is_output"] = True
        self.assertEqual(X3._attrs["shape"], X1._attrs["shape"])

        target = detect_target()
        module = gen_execution_module(
            X3,
            target,
            "./tmp",
            "fused_elementwise_consecutive_1s_broadcast_{}".format(test_name),
        )
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        self.assertEqual(sorted_ops[0]._attrs["op"], "fused_elementwise")
        self.assertEqual(sorted_ops[0]._attrs["read_t"], expected_read_t)
        self.assertEqual(sorted_ops[0]._attrs["op_t"], expected_op_t)
        self.assertEqual(sorted_ops[0]._attrs["data_t"], expected_data_t)

        for k in ks:
            x1_pt = torch.randn(1, 1, k, 1, 1, k).cuda().half()
            x2_pt = torch.randn(1, 1, 1, 1, 1, 1).cuda().half()
            x3_pt = x1_pt / x2_pt
            inputs = {"input0": x1_pt, "input1": x2_pt}
            x3 = torch.empty([1, 1, k, 1, 1, k]).cuda().half()
            module.RunWithTensors(inputs, [x3])
            self.assertTrue(torch.allclose(x3, x3_pt, atol=1e-2, rtol=1e-2))

    def test_consecutive_1s_broadcast(self):
        self._test_consecutive_1s_broadcast(
            ks=[32],
            test_name="static_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )
        self._test_consecutive_1s_broadcast(
            ks=[1, 5, 7, 32],
            test_name="dynamic_shapes",
            expected_read_t="half",
            expected_op_t="half",
            expected_data_t="half",
        )


if __name__ == "__main__":
    unittest.main()
