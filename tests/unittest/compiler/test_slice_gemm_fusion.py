# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils, shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SliceGemmFusionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceGemmFusionTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_slice_gemm_rcr_fusion_a(
        self,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
    ):
        dtype = "float16"

        tensor_B = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_b",
            is_input=True,
        )
        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        Bias = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="bias",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"

        Y = ops.gemm_rcr_bias()(tensor_A, tensor_B, Bias)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        # Run PyTorch
        b_pt = torch.randn(N, K).cuda().half()
        input_pt = torch.randn(*slice_input_shape).cuda().half()
        bias_pt = torch.randn(N).cuda().half()

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        a_pt = input_pt[slice_indices]
        y_pt = torch.nn.functional.linear(a_pt, b_pt, bias=bias_pt)

        # Run AITemplate module.
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors([input_pt, b_pt, bias_pt], [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_gemm_rcr_fusion_a(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # a = [M, K]
        # b = [N, K]
        self._test_slice_gemm_rcr_fusion_a(
            N=4,
            K=8,
            slice_input_shape=(2, 8),
            slice_start_indices=(0, 0),
            slice_end_indices=(None, None),
            test_name="slice_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=32,
            K=6,
            slice_input_shape=(24, 32),
            slice_start_indices=(0, 10),
            slice_end_indices=(None, 16),
            test_name="slice_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=32,
            K=16,
            slice_input_shape=(24, 32),
            slice_start_indices=(0, 2),
            slice_end_indices=(None, 18),
            test_name="slice_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=32,
            K=16,
            slice_input_shape=(24, 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 24),
            test_name="slice_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=32,
            K=16,
            slice_input_shape=(24, 16),
            slice_start_indices=(3, 0),
            slice_end_indices=(15, None),
            test_name="slice_gemm_rcr_fusion_a",
        )

    # This is a test for testing cases where we correctly update a/b_alignment
    # based on input_accessors
    def test_slice_gemm_rcr_fusion_align(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # a = [M, K]
        # b = [N, K]

        # Note that we have to force profiling in ci. Otherwise, we would not
        # be able to fetch cached config.
        target = detect_target()
        old_force_ci = os.environ.get("FORCE_PROFILE", None)
        if target.in_ci_env():
            os.environ["FORCE_PROFILE"] = "1"
        # make a test with smaller alignment
        self._test_slice_gemm_rcr_fusion_a(
            N=3,
            K=16,
            slice_input_shape=(24, 32),
            slice_start_indices=(0, 2),
            slice_end_indices=(None, 18),
            test_name="slice_gemm_rcr_fusion_a",
        )
        # Next, make another one with a larger alignment.
        # If we don't update a/b_alignment accordingly, we would end up with
        # misalignment failures.
        self._test_slice_gemm_rcr_fusion_a(
            N=3,
            K=16,
            slice_input_shape=(24, 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 24),
            test_name="slice_gemm_rcr_fusion_a",
        )

        # another set of tests for a/b alignments
        self._test_slice_gemm_rcr_fusion_b(
            M=21,
            K=4,
            slice_input_shape=(3, 32),
            slice_start_indices=(0, 6),
            slice_end_indices=(None, 10),
            test_name="slice_gemm_rcr_fusion_b",
        )
        self._test_slice_gemm_rcr_fusion_b(
            M=21,
            K=4,
            slice_input_shape=(3, 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 12),
            test_name="slice_gemm_rcr_fusion_b",
        )
        self._test_slice_gemm_rcr_fusion_b(
            M=21,
            K=4,
            slice_input_shape=(3, 32),
            slice_start_indices=(0, 10),
            slice_end_indices=(None, 14),
            test_name="slice_gemm_rcr_fusion_b",
        )

        # another set of tests for a/b alignments
        self._test_slice_gemm_rcr_bias_add(
            M=5,
            N=2,
            K=2,
            slice_input_shape=(5, 32),
            slice_start_indices=(0, 10),
            slice_end_indices=(None, 12),
            test_name="slice_gemm_rcr_bias_add",
        )
        self._test_slice_gemm_rcr_bias_add(
            M=5,
            N=2,
            K=2,
            slice_input_shape=(5, 32),
            slice_start_indices=(0, 16),
            slice_end_indices=(None, 18),
            test_name="slice_gemm_rcr_bias_add",
        )

        # restore old env
        if target.in_ci_env():
            if old_force_ci is None:
                del os.environ["FORCE_PROFILE"]
            else:
                os.environ["FORCE_PROFILE"] = old_force_ci

    def _test_slice_gemm_rcr_fusion_b(
        self,
        M,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
    ):
        dtype = "float16"

        tensor_A = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_a",
            is_input=True,
        )
        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_B = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_B._attrs["name"] = "slice_output"

        Y = ops.gemm_rcr()(tensor_A, tensor_B)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        # Run PyTorch
        a_pt = torch.randn(M, K).cuda().half()
        input_pt = torch.randn(*slice_input_shape).cuda().half()

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        b_pt = input_pt[slice_indices]
        y_pt = torch.nn.functional.linear(a_pt, b_pt)

        # Run AITemplate module.
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors([input_pt, a_pt], [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_gemm_rcr_fusion_b(self):
        # a = [M, K]
        # [slice_end_indices[0] - slice_start_indices[0]] = N
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # b = [N, K]
        self._test_slice_gemm_rcr_fusion_b(
            M=2,
            K=8,
            slice_input_shape=(4, 8),
            slice_start_indices=(0, 0),
            slice_end_indices=(None, None),
            test_name="slice_gemm_rcr_fusion_b",
        )
        self._test_slice_gemm_rcr_fusion_b(
            M=24,
            K=16,
            slice_input_shape=(32, 32),
            slice_start_indices=(0, 16),
            slice_end_indices=(None, 32),
            test_name="slice_gemm_rcr_fusion_b",
        )

    def _test_slice_gemm_rcr_fusion_a_2(
        self,
        M,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
        no_fusion=False,
    ):
        dtype = "float16"

        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        Bias = Tensor(
            shape=[IntImm(M)],
            dtype=dtype,
            name="bias",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"
        a_shape = [d.value() for d in tensor_A._attrs["shape"]]
        assert (
            a_shape[0] == M
        ), f"invalid test shape: expected a_shape[0] to be {M}, but got {a_shape[0]}"
        assert (
            a_shape[1] == M
        ), f"invalid test shape: expected a_shape[1] to be {M}, but got {a_shape[1]}"
        # tensor_A is used for A and B in gemm computation

        Y = ops.gemm_rcr_bias()(tensor_A, tensor_A, Bias)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        if no_fusion:
            self.assertEqual(len(sorted_graph), 4)
        else:
            self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        if no_fusion:
            self.assertEqual(len(sorted_ops), 2)
        else:
            self.assertEqual(len(sorted_ops), 1)

        # Run PyTorch
        input_pt = torch.randn(*slice_input_shape).cuda().half()
        bias_pt = torch.randn(M).cuda().half()

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        a_pt = input_pt[slice_indices]
        y_pt = torch.nn.functional.linear(a_pt, a_pt, bias=bias_pt)

        # Run AITemplate module.
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors([input_pt, bias_pt], [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        dll_name = "test_{}.so".format(self.test_count)
        self.test_count += 1

    def test_slice_gemm_rcr_fusion_a_2(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = M
        # a = [M, M]
        # b = [M, M]
        self._test_slice_gemm_rcr_fusion_a_2(
            M=8,
            slice_input_shape=(8, 24),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 16),
            test_name="slice_gemm_rcr_fusion_a_2",
        )
        self._test_slice_gemm_rcr_fusion_a_2(
            M=8,
            slice_input_shape=(8, 23),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 16),
            test_name="slice_gemm_rcr_fusion_a_2",
            no_fusion=True,
        )

    def _test_slice_gemm_rcr_bias_add(
        self,
        M,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
    ):
        dtype = "float16"

        tensor_B = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_b",
            is_input=True,
        )
        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        Bias = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="bias",
            is_input=True,
        )
        D = Tensor(
            shape=[M, N],
            dtype=dtype,
            name="input_d",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"

        Y1 = ops.gemm_universal.gemm_rcr()(tensor_A, tensor_B)
        Y2 = ops.elementwise(FuncEnum.ADD)(Y1, Bias)
        Y = ops.elementwise(FuncEnum.ADD)(Y2, D)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 5)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        # Run PyTorch
        b_pt = torch.randn(N, K).cuda().half()
        input_pt = torch.randn(*slice_input_shape).cuda().half()
        bias_pt = torch.randn(N).cuda().half()
        d_pt = torch.randn(M, N).cuda().half()

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        a_pt = input_pt[slice_indices]
        y2_pt = torch.nn.functional.linear(a_pt, b_pt, bias=bias_pt)
        y_pt = y2_pt + d_pt

        # Run AITemplate module.
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors([input_pt, b_pt, bias_pt, d_pt], [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_gemm_rcr_bias_add(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # a = [M, K]
        # b = [N, K]
        self._test_slice_gemm_rcr_bias_add(
            M=4,
            N=2,
            K=8,
            slice_input_shape=(4, 16),
            slice_start_indices=(0, 0),
            slice_end_indices=(None, 8),
            test_name="slice_gemm_rcr_bias_add",
        )
        self._test_slice_gemm_rcr_bias_add(
            M=4,
            N=2,
            K=8,
            slice_input_shape=(4, 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 16),
            test_name="slice_gemm_rcr_bias_add",
        )

    def test_slice_nd_gemm_rcr_fusion_a(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M0
        # [slice_end_indices[1] - slice_start_indices[1]] = M1
        # ...
        # [slice_end_indices[2] - slice_start_indices[2]] = K
        # a = [M0, M1, ..., K]
        # b = [N, K]
        self._test_slice_gemm_rcr_fusion_a(
            N=5,
            K=8,
            slice_input_shape=(4, 13, 2, 32),
            slice_start_indices=(0, 0, 0, 8),
            slice_end_indices=(None, None, None, 16),
            test_name="slice_nd_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=5,
            K=4,
            slice_input_shape=(4, 13, 2, 32),
            slice_start_indices=(0, 0, 0, 8),
            slice_end_indices=(None, None, None, 12),
            test_name="slice_nd_gemm_rcr_fusion_a",
        )
        self._test_slice_gemm_rcr_fusion_a(
            N=5,
            K=4,
            slice_input_shape=(13, 2, 32),
            slice_start_indices=(0, 0, 10),
            slice_end_indices=(None, None, 14),
            test_name="slice_nd_gemm_rcr_fusion_a",
        )

    def _test_slice_gemm_rcr_fusion_dynamic(
        self,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
    ):
        dtype = "float16"

        tensor_B = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_b",
            is_input=True,
        )
        x_shape = [
            shape_utils.gen_int_var_min_max(d) if isinstance(d, list) else IntImm(d)
            for d in slice_input_shape
        ]
        X = Tensor(
            shape=x_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        Bias = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="bias",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"

        Y = ops.gemm_rcr_bias()(tensor_A, tensor_B, Bias)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        Ms = None
        for d in slice_input_shape:
            if isinstance(d, list):
                Ms = d
                break
        assert Ms is not None, "expected to have at least one dynamic dim"
        for idx in range(len(Ms)):
            # Run PyTorch
            b_pt = torch.randn(N, K).cuda().half()
            input_shape_pt = [
                d[idx] if isinstance(d, list) else d for d in slice_input_shape
            ]
            input_pt = torch.randn(*input_shape_pt).cuda().half()
            bias_pt = torch.randn(N).cuda().half()

            slice_indices = [
                slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
            ]
            a_pt = input_pt[slice_indices]
            y_pt = torch.nn.functional.linear(a_pt, b_pt, bias=bias_pt)

            # Run AITemplate module.
            y = torch.empty(y_pt.size()).cuda().half()
            module.RunWithTensors([input_pt, b_pt, bias_pt], [y])
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self.test_count += 1

    def test_slice_gemm_rcr_fusion_dynamic(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # a = [M, K]
        # b = [N, K]
        self._test_slice_gemm_rcr_fusion_dynamic(
            N=4,
            K=8,
            slice_input_shape=([4, 9], 8),
            slice_start_indices=(0, 0),
            slice_end_indices=(None, None),
            test_name="slice_gemm_rcr_fusion_dynamic",
        )
        self._test_slice_gemm_rcr_fusion_dynamic(
            N=4,
            K=8,
            slice_input_shape=([4, 9], 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 16),
            test_name="slice_gemm_rcr_fusion_dynamic",
        )
        self._test_slice_gemm_rcr_fusion_dynamic(
            N=4,
            K=8,
            slice_input_shape=([10, 20], [4, 9], 32),
            slice_start_indices=(0, 0, 8),
            slice_end_indices=(None, None, 16),
            test_name="slice_gemm_rcr_fusion_dynamic",
        )

    def _test_slice_multiple_gemm_rcr_fusion_a(
        self,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        test_name,
    ):
        dtype = "float16"

        tensor_B1 = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_b1",
            is_input=True,
        )
        tensor_B2 = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_b2",
            is_input=True,
        )
        X = Tensor(
            shape=slice_input_shape,
            dtype=dtype,
            name="input_x",
            is_input=True,
        )
        Bias = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="bias",
            is_input=True,
        )

        slice_op = ops.dynamic_slice()
        tensor_A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        tensor_A._attrs["name"] = "slice_output"

        Y1 = ops.gemm_rcr_bias()(tensor_A, tensor_B1, Bias)
        Y2 = ops.gemm_rcr_bias()(tensor_A, tensor_B2, Bias)
        Y = ops.elementwise(FuncEnum.ADD)(Y1, Y2)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = gen_execution_module(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 6)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Run PyTorch
        b1_pt = torch.randn(N, K).cuda().half()
        b2_pt = torch.randn(N, K).cuda().half()
        input_pt = torch.randn(*slice_input_shape).cuda().half()
        bias_pt = torch.randn(N).cuda().half()

        slice_indices = [
            slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
        ]
        a_pt = input_pt[slice_indices]
        y1_pt = torch.nn.functional.linear(a_pt, b1_pt, bias=bias_pt)
        y2_pt = torch.nn.functional.linear(a_pt, b2_pt, bias=bias_pt)
        y_pt = y1_pt + y2_pt

        # Run AITemplate module.
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors(
            {
                "input_x": input_pt,
                "input_b1": b1_pt,
                "input_b2": b2_pt,
                "bias": bias_pt,
            },
            [y],
        )
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_multiple_gemm_rcr_fusion_a(self):
        # [slice_end_indices[0] - slice_start_indices[0]] = M
        # [slice_end_indices[1] - slice_start_indices[1]] = K
        # a = [M, K]
        # b = [N, K]
        self._test_slice_multiple_gemm_rcr_fusion_a(
            N=4,
            K=16,
            slice_input_shape=(30, 32),
            slice_start_indices=(0, 8),
            slice_end_indices=(None, 24),
            test_name="slice_multiple_gemm_rcr_fusion_a",
        )
        self._test_slice_multiple_gemm_rcr_fusion_a(
            N=4,
            K=6,
            slice_input_shape=(30, 32),
            slice_start_indices=(0, 12),
            slice_end_indices=(None, 18),
            test_name="slice_multiple_gemm_rcr_fusion_a",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
