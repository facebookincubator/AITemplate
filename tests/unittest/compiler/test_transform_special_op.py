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

from typing import List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import shape_utils
from aitemplate.utils.graph_utils import get_sorted_ops

from parameterized import parameterized


class GemmRrrSmallNkTestCase(unittest.TestCase):
    def _create_gemm_rrr_graph(self, M, K, N, dtype):
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[K, N], dtype=dtype, name="input_1", is_input=True)
        OP = ops.gemm_rrr()
        Y = OP(X, W)
        Y._attrs["name"] = "gemm_rrr_tensor"
        Y._attrs["is_output"] = True

        return X, W, Y

    def _test_small_nk(self, Ms, N, K, testname=None, dtype="float16"):
        if testname is None:
            testname = f"gemm_rrr_small_nk_{Ms}_{N}_{K}_{dtype}"
            testname = testname.replace(" ", "")
            testname = testname.replace("[", "")
            testname = testname.replace("]", "")

        X, W, gemm_tensor = self._create_gemm_rrr_graph(
            shape_utils.gen_int_var_min_max(Ms), K, N, dtype
        )

        output = ops.elementwise(FuncEnum.COS)(gemm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model([output, gemm_tensor], target, "./tmp", testname)

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
            X_pt = get_random_torch_tensor([m, K], dtype)
            W_pt = get_random_torch_tensor([K, N], dtype)
            mm_pt = torch.matmul(X_pt, W_pt)
            Y_pt = torch.cos(mm_pt)
            y = get_torch_empty_tensor([m, N], dtype)
            gemm_tensor_pt = get_torch_empty_tensor([m, N], dtype)
            module.run_with_tensors(
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

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_small_nk_fp32(self):
        self._test_small_nk([10], 8, 4, "test_small_nk_fp32", dtype="float32")
        self._test_small_nk(
            [10, 30, 50], 6, 4, "test_small_kn_dynamic1_fp32", dtype="float32"
        )
        self._test_small_nk(
            [100, 200], 6, 3, "test_small_nk_alignment_fp32", dtype="float32"
        )

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_small_nk_no_transform(self, dtype):
        target = detect_target()

        M, K, N = 8, 8, 16
        _, _, output = self._create_gemm_rrr_graph(M, K, N, dtype)

        module = compile_model(
            output, target, "./tmp", f"test_small_nk_fail_{M}_{K}_{N}_{dtype}"
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

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([K, N], dtype)
        Y_pt = torch.matmul(X_pt, W_pt)

        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


class BmmRcrN1TestCase(unittest.TestCase):
    def _create_bmm_rcr_graph(self, B, M, N, K, dtype):
        X = Tensor(shape=[B, M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="input_1", is_input=True)
        OP = ops.bmm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "bmm_rcr_tensor"

        return X, W, Y

    def _test_n1_k8(self, B, M, N, K, testname=None, dtype="float16"):
        if testname is None:
            testname = f"bmm_rcr_n1_{B}_{M}_{N}_{K}_{dtype}"
            testname = testname.replace(" ", "")
            testname = testname.replace("[", "")
            testname = testname.replace("]", "")

        X, W, bmm_tensor = self._create_bmm_rcr_graph(
            B, shape_utils.gen_int_var_min_max(M), N, K, dtype
        )
        mul = ops.elementwise(FuncEnum.MUL)(
            bmm_tensor, Tensor(shape=[], dtype=dtype, value=1.0)
        )
        output = ops.elementwise(FuncEnum.COS)(mul)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

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
            X_pt = get_random_torch_tensor([B, m, K], dtype)
            W_pt = get_random_torch_tensor([B, N, K], dtype)

            def pt_bmm(X_pt, W_pt):
                WT = torch.transpose(W_pt, 2, 1)
                Y_pt = torch.bmm(X_pt, WT)
                return Y_pt

            Y_pt = torch.cos(pt_bmm(X_pt, W_pt))

            y = get_torch_empty_tensor([B, m, N], dtype)
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_n1_k8(self):
        self._test_n1_k8(1, [8], 1, 8)
        self._test_n1_k8(10, [8], 1, 8)

    def test_n1_k8_dynamic(self):
        self._test_n1_k8(10, [8, 16], 1, 8)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_n1_k8_fp32(self):
        self._test_n1_k8(10, [8], 1, 8, dtype="float32")
        self._test_n1_k8(10, [8, 16], 1, 8, dtype="float32")

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_n_non1_fail(self, dtype):
        target = detect_target()

        B, M, K, N = 8, 8, 8, 8
        _, _, output = self._create_bmm_rcr_graph(B, M, K, N, dtype)
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", f"bmm_rcr_n_non1_{dtype}")

        output_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_rcr_tensor":
                output_tensor = tensor
                break

        self.assertIsNotNone(output_tensor, "bmm_rcr tensor not found")
        self.assertEqual(len(output_tensor._attrs["src_ops"]), 1)
        src_op = next(iter(output_tensor._attrs["src_ops"]))
        self.assertEqual(src_op._attrs["op"], "bmm_rcr")


class OneByOneConvTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter = 0

    def _assert_no_convs(self, sorted_graph: List[Tensor]):
        for op in get_sorted_ops(sorted_graph):
            self.assertFalse(op._attrs["op"].startswith("conv2d"))

    def _assert_has_gemm(self, sorted_graph: List[Tensor]):
        for op in get_sorted_ops(sorted_graph):
            if op._attrs["op"].startswith("gemm_rcr"):
                return
        raise AssertionError("Did not find gemm_rcr in graph")

    def _test_simple_1x1_conv(
        self, batch, CO, HH, WW, CI, activation=None, with_bias=False, dtype="float16"
    ):
        if isinstance(batch, int):
            batch = (batch,)
        batch_var = shape_utils.gen_int_var_min_max(batch, name="batch_size")
        X = Tensor(
            shape=[batch_var, HH, WW, CI],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[CO, 1, 1, CI],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )

        if with_bias:
            bias = Tensor(
                shape=[CO],
                dtype=dtype,
                name="bias",
                is_input=True,
            )
            conv2d = ops.conv2d_bias(stride=1, pad=0)(X, W, bias)
        else:
            conv2d = ops.conv2d(stride=1, pad=0)(X, W)

        if activation == "relu":
            conv2d = ops.elementwise(FuncEnum.RELU)(conv2d)
        elif activation == "sigmoid":
            conv2d = ops.elementwise(FuncEnum.SIGMOID)(conv2d)

        elif activation == "hardswish":
            # We have no FuncEnum.HARDSWISH, must use fused version
            if with_bias:
                conv2d = ops.conv2d_bias_hardswish(stride=1, pad=0)(X, W, bias)
            else:
                raise NotImplementedError("Cannot use hardswish on conv2d without bias")

        elif activation is not None:
            raise NotImplementedError(f"Unsupported activation {activation}")

        conv2d._attrs["name"] = "output"
        conv2d._attrs["is_output"] = True

        with compile_model(
            conv2d,
            detect_target(),
            "./tmp",
            f"test_simple_one_by_one_conv_{self._counter}",
        ) as module:
            self._counter += 1
            self._assert_no_convs(module.debug_sorted_graph)
            self._assert_has_gemm(module.debug_sorted_graph)

            for batch_pt in batch:
                X_pt = get_random_torch_tensor([batch_pt, CI, HH, WW], dtype)
                W_pt = get_random_torch_tensor([CO, CI, 1, 1], dtype)

                if with_bias:
                    B_pt = get_random_torch_tensor([CO], dtype)
                else:
                    B_pt = None

                Y_pt = torch.nn.functional.conv2d(
                    X_pt, W_pt, bias=B_pt, stride=1, padding=0
                )

                if activation == "relu":
                    Y_pt = torch.relu(Y_pt)
                elif activation == "sigmoid":
                    Y_pt = torch.sigmoid(Y_pt)
                elif activation == "hardswish":
                    Y_pt = torch.nn.functional.hardswish(Y_pt)
                elif activation is not None:
                    raise NotImplementedError(f"Unsupported activation {activation}")

                Y_ait = get_torch_empty_tensor(batch_pt, HH, WW, CO, dtype)
                inputs = {
                    "input_0": X_pt.permute(0, 2, 3, 1).contiguous(),
                    "input_1": W_pt.permute(0, 2, 3, 1).contiguous(),
                }
                if with_bias:
                    inputs["bias"] = B_pt

                module.run_with_tensors(inputs, {"output": Y_ait})

                torch.testing.assert_close(
                    Y_pt, Y_ait.permute(0, 3, 1, 2), atol=1e-1, rtol=1e-1
                )

    # !!! SKIPPED TESTS BELOW !!!
    # TODO: enable the tests when ck is fixed

    # def test_1x1_conv_no_bias(self):
    #     self._test_simple_1x1_conv(batch=1, CO=256, HH=3, WW=4, CI=2)
    #     self._test_simple_1x1_conv(
    #         batch=3, CO=100, HH=200, WW=4, CI=2, activation="relu"
    #     )
    #     self._test_simple_1x1_conv(
    #         batch=2, CO=128, HH=10, WW=42, CI=3, activation="sigmoid"
    #     )
    #     self._test_simple_1x1_conv(batch=5, CO=256, HH=15, WW=5, CI=13)
    #     self._test_simple_1x1_conv(batch=(1, 10), CO=128, HH=2, WW=2, CI=10)

    # def test_1x1_conv_with_bias(self):
    #     self._test_simple_1x1_conv(batch=1, CO=256, HH=3, WW=4, CI=2, with_bias=True)
    #     self._test_simple_1x1_conv(
    #         batch=3,
    #         CO=100,
    #         HH=200,
    #         WW=4,
    #         CI=2,
    #         activation="relu",
    #         with_bias=True,
    #     )
    #     self._test_simple_1x1_conv(
    #         batch=2, CO=128, HH=10, WW=42, CI=3, activation="sigmoid", with_bias=True
    #     )
    #     self._test_simple_1x1_conv(
    #         batch=2, CO=64, HH=10, WW=42, CI=3, activation="hardswish", with_bias=True
    #     )
    #     self._test_simple_1x1_conv(batch=5, CO=256, HH=15, WW=5, CI=13, with_bias=True)
    #     self._test_simple_1x1_conv(
    #         batch=(1, 10), CO=128, HH=2, WW=2, CI=10, with_bias=True
    #     )


if __name__ == "__main__":
    unittest.main()
