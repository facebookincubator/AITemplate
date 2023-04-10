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

from typing import Sequence

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.fuse_parallel_gemms import fuse_parallel_gemms
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    count_ops,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    has_op,
)
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


def _prepare_input_tensors(m, nk_groups, dtype, start=0, has_bias=True):
    inputs = []
    batch_dim = IntImm(m)
    for i, (n, k) in enumerate(nk_groups):
        X = Tensor(
            shape=[batch_dim, IntImm(k)],
            dtype=dtype,
            name="x_{}".format(i + start),
            is_input=True,
        )
        W = Tensor(
            shape=[IntImm(n), IntImm(k)],
            dtype=dtype,
            name="w_{}".format(i + start),
        )
        B = Tensor(
            shape=[IntImm(n)],
            dtype=dtype,
            name="b_{}".format(i + start),
        )
        if has_bias:
            inputs.append([X, W, B])
        else:
            inputs.append([X, W])
    return inputs


def _prepare_inputs_and_constants(m, nk_groups, dtype, start=0, has_bias=True):
    inputs = []
    constants = {}

    for i, (n, k) in enumerate(nk_groups):
        x_pt = get_random_torch_tensor([m, k], dtype)
        w_pt = get_random_torch_tensor([n, k], dtype)
        b_pt = get_random_torch_tensor([n], dtype)

        inputs.append(x_pt)
        constants[f"w_{i}"] = w_pt
        if has_bias:
            constants[f"b_{i}"] = b_pt

    return inputs, constants


def _prepare_outputs(output_tensors, dtype):
    def _to_int_list(shape):
        result = []
        for d in shape:
            assert isinstance(d, IntImm)
            result.append(d._attrs["values"][0])
        return result

    output_shapes = [_to_int_list(output._attrs["shape"]) for output in output_tensors]
    outputs = [get_torch_empty_tensor(shape, dtype) for shape in output_shapes]
    return outputs


def _prepare_ait_module(m, nk_groups, constants, dtype, test_idx=0, has_bias=True):
    group_input_tensors = _prepare_input_tensors(m, nk_groups, dtype, has_bias=has_bias)
    output_tensors = []
    for group in group_input_tensors:
        group[0] = ops.elementwise(FuncEnum.TANH)(group[0])
        Y = ops.gemm_rcr_bias()(*group) if has_bias else ops.gemm_rcr()(*group)
        output_tensors.append(Y)

    Y = ops.concatenate()(output_tensors, dim=-1)
    Y._attrs["name"] = "y"
    Y._attrs["is_output"] = True

    target = detect_target()
    module = compile_model(
        Y,
        target,
        "./tmp",
        f"test_multi_parallel_gemm_cat_groups_{dtype}",
        dll_name=f"test_{test_idx}.so",
        constants=constants,
    )
    outputs = _prepare_outputs([Y], dtype)
    return outputs, module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ParallelGemmCatFusionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParallelGemmCatFusionTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _fuse_2_split_parallel_gemm_cat(
        self, b: int, ms: Sequence[int], n: int, k: int, dtype: str = "float16"
    ):
        _LOGGER.info(
            f"_fuse_2_split_parallel_gemm_cat, b: {b}, ms: {ms}, n: {n}, k: {k}",
        )
        X1 = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype=dtype,
            name="X1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(2 * b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype=dtype,
                name=f"W{i}",
                is_input=True,
            )
            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype=dtype,
                name=f"B{i}",
                is_input=True,
            )
            Bs.append(B)

        X3 = ops.split()(X1, k, dim=-1)
        X4 = ops.split()(X2, k, dim=-1)
        cat_inputs = []
        gemm_inputs = X3 + X4
        for i in range(2 * b):
            X5 = ops.gemm_rcr_bias()(gemm_inputs[i], Ws[i], Bs[i])
            cat_inputs.append(X5)
        cat_output = ops.concatenate()(cat_inputs, dim=-1)

        cat_output._attrs["name"] = "output0"
        cat_output._attrs["is_output"] = True

        sorted_graph = toposort(cat_output)
        new_sorted_graph = fuse_parallel_gemms(sorted_graph)

        sorted_ops = graph_utils.get_sorted_ops(new_sorted_graph)
        assert not has_op(
            sorted_ops, "perm102_bmm_rrr_bias"
        ), "the final graph should not have op perm102_bmm_rrr_bias"
        assert not has_op(
            sorted_ops, "perm102_bmm_rcr_bias"
        ), "the final graph should not have op perm102_bmm_rcr_bias"

    def _fuse_parallel_gemm_cat(
        self,
        b: int,
        ms: Sequence[int],
        n: int,
        k: int,
        perm102_bmm_op: str,
        has_tanh: bool = True,
        reshape_weight: bool = False,
        dtype: str = "float16",
    ):
        _LOGGER.info(f"_fuse_parallel_gemm_cat, b: {b}, ms: {ms}, n: {n}, k: {k}")
        X = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype=dtype,
                name=f"W{i}",
            )
            if reshape_weight:
                W = ops.reshape()(W, [n, k])  # no-op, for testing
            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype=dtype,
                name=f"B{i}",
            )
            Bs.append(B)

        X1 = ops.split()(X, k, dim=-1)
        cat_inputs = []
        for i in range(b):
            X2 = ops.elementwise(FuncEnum.TANH)(X1[i]) if has_tanh else X1[i]
            X3 = ops.gemm_rcr_bias()(X2, Ws[i], Bs[i])
            cat_inputs.append(X3)
        cat_output = ops.concatenate()(cat_inputs, dim=-1)

        cat_output._attrs["name"] = "output0"
        cat_output._attrs["is_output"] = True

        constants = {}
        for i in range(b):
            constants[f"W{i}"] = get_random_torch_tensor([n, k], dtype)
            constants[f"B{i}"] = get_random_torch_tensor([n], dtype)

        # Gen module.
        target = detect_target()
        with compile_model(
            [cat_output],
            target,
            "./tmp",
            f"fuse_parallel_gemm_cat_{dtype}",
            dll_name=f"test_{self._test_id}.so",
            constants=constants,
        ) as module:
            self._test_id += 1
            # Verify the generated graph.
            sorted_graph = module.debug_sorted_graph
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            assert has_op(
                sorted_ops, perm102_bmm_op
            ), f"the final graph does not have op {perm102_bmm_op}"
            if not has_tanh:
                assert not has_op(
                    sorted_ops, "split"
                ), "the final graph has split op, but it should not"

            for m in ms:
                x_pt = get_random_torch_tensor([m, b * k], dtype)
                x1_pt = torch.split(x_pt, k, dim=-1)

                cat_inputs_pt = []
                for i in range(b):
                    x2_pt = x1_pt[i].tanh() if has_tanh else x1_pt[i]
                    x3_pt = torch.nn.functional.linear(
                        x2_pt, constants[f"W{i}"], constants[f"B{i}"]
                    )
                    cat_inputs_pt.append(x3_pt)
                cat_output_pt = torch.cat(cat_inputs_pt, dim=-1)

                # Run AITemplate module.

                out = get_torch_empty_tensor([m, b * n], dtype)
                module.run_with_tensors([x_pt], [out])
                # module.benchmark_with_tensors([x_pt], [out])

                # Do comparisons.
                self.assertTrue(
                    torch.allclose(out, cat_output_pt, atol=5e-2, rtol=5e-2)
                )

    def test_fuse_parallel_gemm_cat_fp16(self):
        # test n x gemms + cat
        self._fuse_parallel_gemm_cat(
            b=4, ms=[256, 512], n=128, k=64, perm102_bmm_op="perm102_bmm_rrr_bias"
        )
        self._fuse_parallel_gemm_cat(
            b=4, ms=[256, 512], n=128, k=100, perm102_bmm_op="perm102_bmm_rrr_bias"
        )
        self._fuse_parallel_gemm_cat(
            b=4, ms=[128, 256], n=100, k=32, perm102_bmm_op="perm102_bmm_rcr_bias"
        )
        self._fuse_parallel_gemm_cat(
            b=16, ms=[15, 31], n=7, k=5, perm102_bmm_op="perm102_bmm_rrr_bias"
        )
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[128, 256],
            n=100,
            k=32,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            reshape_weight=True,
        )

        # test split + n x gemms + cat
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[256, 512],
            n=128,
            k=64,
            perm102_bmm_op="perm102_bmm_rrr_bias",
            has_tanh=False,
        )
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[128, 256],
            n=100,
            k=32,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            has_tanh=False,
        )
        self._fuse_parallel_gemm_cat(
            b=16,
            ms=[15, 31],
            n=7,
            k=5,
            perm102_bmm_op="perm102_bmm_rrr_bias",
            has_tanh=False,
        )
        self._fuse_parallel_gemm_cat(
            b=16,
            ms=[1024, 2048],
            n=100,
            k=128,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            has_tanh=False,
        )

        # test multiple split + n x gemms + cat
        self._fuse_2_split_parallel_gemm_cat(b=4, ms=[256, 512], n=128, k=64)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_fuse_parallel_gemm_cat_fp32_sm80(self):
        # test n x gemms + cat
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[256, 512],
            n=128,
            k=64,
            perm102_bmm_op="perm102_bmm_rrr_bias",
            dtype="float32",
        )
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[128, 256],
            n=10,
            k=32,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            dtype="float32",
        )
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[128, 256],
            n=10,
            k=32,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            reshape_weight=True,
            dtype="float32",
        )

        # test split + n x gemms + cat
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[256, 512],
            n=32,
            k=64,
            perm102_bmm_op="perm102_bmm_rrr_bias",
            has_tanh=False,
            dtype="float32",
        )
        self._fuse_parallel_gemm_cat(
            b=4,
            ms=[128, 256],
            n=10,
            k=32,
            perm102_bmm_op="perm102_bmm_rcr_bias",
            has_tanh=False,
            dtype="float32",
        )

        # test multiple split + n x gemms + cat
        self._fuse_2_split_parallel_gemm_cat(
            b=4, ms=[256, 512], n=128, k=64, dtype="float32"
        )

    def _test_fuse_parallel_gemm_cat_partial(
        self,
        b1: int,
        b2: int,
        ms: Sequence[int],
        n: int,
        k: int,
        has_tanh: bool = True,
        dtype: str = "float16",
    ):
        _LOGGER.info(
            f"_fuse_parallel_gemm_cat_partial, b1: {b1}, b2: {b2}, ms: {ms}, n: {n}, k: {k}",
        )
        batch_dim = IntVar(ms, "input_batch")
        b = b1 + b2
        X1 = Tensor(
            shape=[batch_dim, IntImm(b1 * k)],
            dtype=dtype,
            name="X1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_dim, IntImm(b2 * k)],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype=dtype,
                name=f"W{i}",
            )
            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype=dtype,
                name=f"B{i}",
            )
            Bs.append(B)

        cat_inputs = []

        X3 = ops.split()(X1, k, dim=-1)
        for i in range(b1):
            X5 = ops.elementwise(FuncEnum.TANH)(X3[i]) if has_tanh else X3[i]
            X6 = ops.gemm_rcr_bias()(X5, Ws[i], Bs[i])
            cat_inputs.append(X6)

        X7 = ops.reshape()(X1, [-1, b1, k])
        W = Tensor(
            shape=[IntImm(b1), IntImm(n), IntImm(k)],
            dtype=dtype,
            name="W",
        )
        B = Tensor(
            shape=[IntImm(b1), IntImm(n)],
            dtype=dtype,
            name="B",
        )
        WT = ops.permute021()(W)

        X8 = ops.perm102_bmm_rcr()(X7, W)
        X9 = ops.reshape()(X8, [batch_dim, -1])
        cat_inputs.append(X9)

        X10 = ops.perm102_bmm_rcr_bias()(X7, W, B)
        X11 = ops.reshape()(X10, [batch_dim, -1])
        cat_inputs.append(X11)

        X12 = ops.perm102_bmm_rrr()(X7, WT)
        X13 = ops.reshape()(X12, [batch_dim, -1])
        cat_inputs.append(X13)

        X14 = ops.perm102_bmm_rrr_bias()(X7, WT, B)
        X15 = ops.reshape()(X14, [batch_dim, -1])
        cat_inputs.append(X15)

        X4 = ops.split()(X2, k, dim=-1)
        for i in range(b2):
            X5 = ops.elementwise(FuncEnum.TANH)(X4[i]) if has_tanh else X4[i]
            X6 = ops.gemm_rcr_bias()(X5, Ws[i + b1], Bs[i + b1])
            cat_inputs.append(X6)

        cat_output = ops.concatenate()(cat_inputs, dim=-1)

        cat_output._attrs["name"] = "output0"
        cat_output._attrs["is_output"] = True

        constants = {}
        for i in range(b):
            constants[f"W{i}"] = get_random_torch_tensor([n, k], dtype)
            constants[f"B{i}"] = get_random_torch_tensor([n], dtype)

        constants["W"] = get_random_torch_tensor([b1, n, k], dtype)
        constants["B"] = get_random_torch_tensor([b1, n], dtype)

        # Gen module.
        target = detect_target()
        with compile_model(
            [cat_output],
            target,
            "./tmp",
            f"fuse_parallel_gemm_cat_{dtype}",
            dll_name=f"test_{self._test_id}.so",
            constants=constants,
        ) as module:
            self._test_id += 1
            # Verify the generated graph.
            sorted_graph = module.debug_sorted_graph
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            assert not has_op(
                sorted_ops, "gemm_rcr_bias"
            ), "the final graph still has op gemm_rcr_bias"
            if not has_tanh:
                assert not has_op(
                    sorted_ops, "split"
                ), "the final graph has split op, but it should not"

            for m in ms:
                x_pt = get_random_torch_tensor([m, b1 * k], dtype)
                x1_pt = torch.split(x_pt, k, dim=-1)

                cat_inputs_pt = []
                for i in range(b1):
                    x3_pt = x1_pt[i].tanh() if has_tanh else x1_pt[i]
                    x4_pt = torch.nn.functional.linear(
                        x3_pt, constants[f"W{i}"], constants[f"B{i}"]
                    )
                    cat_inputs_pt.append(x4_pt)

                x5_pt = x_pt.reshape(m, b1, k).permute([1, 0, 2])  # [b, m, k]
                # [b, m, k] x [b, k, n] -> [b, m, n]
                x6_pt = torch.bmm(x5_pt, constants["W"].permute([0, 2, 1]))
                x7_pt = x6_pt.permute([1, 0, 2])  # [m, b, n]
                x8_pt = x7_pt.reshape([m, -1])  # [m, b * n]
                cat_inputs_pt.append(x8_pt)

                x9_pt = (x7_pt + constants["B"]).reshape([m, -1])
                cat_inputs_pt.append(x9_pt)
                cat_inputs_pt.append(x8_pt)
                cat_inputs_pt.append(x9_pt)

                xx_pt = get_random_torch_tensor([m, b2 * k], dtype)
                x2_pt = torch.split(xx_pt, k, dim=-1)
                for i in range(b2):
                    x3_pt = x2_pt[i].tanh() if has_tanh else x2_pt[i]
                    x4_pt = torch.nn.functional.linear(
                        x3_pt, constants[f"W{i + b1}"], constants[f"B{i + b1}"]
                    )
                    cat_inputs_pt.append(x4_pt)

                cat_output_pt = torch.cat(cat_inputs_pt, dim=-1)

                # Run AITemplate module.

                out = get_torch_empty_tensor(cat_output_pt.size(), dtype)
                module.run_with_tensors({"X1": x_pt, "X2": xx_pt}, {"output0": out})

                # Do comparisons.
                self.assertTrue(
                    torch.allclose(out, cat_output_pt, atol=5e-2, rtol=5e-2)
                )

    def test_fuse_parallel_gemm_cat_partial_fp16(self):
        self._test_fuse_parallel_gemm_cat_partial(4, 4, [128, 256], 32, 64, True)
        self._test_fuse_parallel_gemm_cat_partial(4, 4, [128, 256], 32, 64, False)
        self._test_fuse_parallel_gemm_cat_partial(3, 3, [128, 256], 30, 66, True)
        self._test_fuse_parallel_gemm_cat_partial(2, 2, [128, 256], 33, 55, True)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_fuse_parallel_gemm_cat_partial_fp32_sm80(self):
        self._test_fuse_parallel_gemm_cat_partial(
            4, 4, [128, 256], 32, 64, True, dtype="float32"
        )
        self._test_fuse_parallel_gemm_cat_partial(
            4, 4, [128, 256], 32, 64, False, dtype="float32"
        )

    def _test_multi_parallel_gemm_cat_groups(
        self, m, nk_groups, num_unfused_ops=0, dtype="float16"
    ):
        inputs, constants = _prepare_inputs_and_constants(m, nk_groups, dtype)
        outputs, module = _prepare_ait_module(
            m, nk_groups, constants, dtype, test_idx=self._test_id
        )
        self._test_id += 1
        with module:
            sorted_graph = module.debug_sorted_graph
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            actual_unfused_ops = count_ops(sorted_ops, "gemm_rcr_bias")
            assert (
                actual_unfused_ops == num_unfused_ops
            ), f"Expecting {num_unfused_ops} unfused gemm_rcr_bias ops, found {actual_unfused_ops}"
            ys = []
            for i, input in enumerate(inputs):
                tanh = input.tanh()
                y = torch.nn.functional.linear(
                    tanh, constants[f"w_{i}"], constants[f"b_{i}"]
                )
                ys.append(y)
            pt_y = torch.cat(ys, dim=-1)
            module.run_with_tensors(inputs, outputs)
            self.assertTrue(torch.allclose(pt_y, outputs[0], atol=5e-2, rtol=5e-2))

    def test_multi_parallel_gemm_cat_groups_fp16(self):
        self._test_multi_parallel_gemm_cat_groups(
            256,
            [[128, 64]] * 2 + [[128, 120]] * 4 + [[128, 72]] * 2 + [[128, 64]] * 2,
        )
        self._test_multi_parallel_gemm_cat_groups(
            256, [[128, 64]] * 2 + [[128, 120]] + [[128, 72]] * 2 + [[128, 64]], 2
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_multi_parallel_gemm_cat_groups_fp32_sm80(self):
        self._test_multi_parallel_gemm_cat_groups(
            256,
            [[128, 64]] * 2 + [[128, 120]] * 4 + [[128, 72]] * 2 + [[128, 64]] * 2,
            dtype="float32",
        )

    def _skip_fuse_parallel_gemm_output_cat(
        self,
        b: int,
        ms: Sequence[int],
        n: int,
        k: int,
        perm102_bmm_op: str,
        dtype: str = "float16",
    ):
        _LOGGER.info(f"_skip_fuse_parallel_gemm_cat, b: {b}, ms: {ms}, n: {n}, k: {k}")
        X = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype=dtype,
                name=f"W{i}",
            )

            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype=dtype,
                name=f"B{i}",
            )
            Bs.append(B)

        X1 = ops.split()(X, k, dim=-1)
        cat_inputs = []
        for i in range(b):
            X2 = X1[i]
            X3 = ops.gemm_rcr_bias()(X2, Ws[i], Bs[i])
            cat_inputs.append(X3)
            X3._attrs["name"] = f"output{i+1}"
            X3._attrs["is_output"] = True

        cat_output = ops.concatenate()(cat_inputs, dim=-1)

        cat_output._attrs["name"] = "output0"
        cat_output._attrs["is_output"] = True

        constants = {}
        for i in range(b):
            constants[f"W{i}"] = get_random_torch_tensor([n, k], dtype)
            constants[f"B{i}"] = get_random_torch_tensor([n], dtype)

        # Gen module.
        target = detect_target()
        with compile_model(
            [cat_output, *cat_inputs],
            target,
            "./tmp",
            f"fuse_parallel_gemm_cat_{dtype}",
            dll_name=f"test_{self._test_id}.so",
            constants=constants,
        ) as module:
            self._test_id += 1
            # Verify the generated graph.
            sorted_graph = module.debug_sorted_graph
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            assert not has_op(
                sorted_ops, perm102_bmm_op
            ), f"the final graph has op {perm102_bmm_op}"
            assert has_op(
                sorted_ops, "gemm_rcr_bias"
            ), "the final graph does not have op gemm_rcr_bias"

            for m in ms:
                x_pt = get_random_torch_tensor([m, b * k], dtype)
                x1_pt = torch.split(x_pt, k, dim=-1)

                cat_inputs_pt = []
                for i in range(b):
                    x2_pt = x1_pt[i]
                    x3_pt = torch.nn.functional.linear(
                        x2_pt, constants[f"W{i}"], constants[f"B{i}"]
                    )
                    cat_inputs_pt.append(x3_pt)
                cat_output_pt = (torch.cat(cat_inputs_pt, dim=-1), *cat_inputs_pt)

                # Run AITemplate module.

                cat_out = get_torch_empty_tensor([m, b * n], dtype)
                out_other = [
                    get_torch_empty_tensor(x.shape, dtype) for x in cat_inputs_pt
                ]
                out = [cat_out, *out_other]
                module.run_with_tensors([x_pt], out)

                # Do comparisons.
                for (out_ait, out_pt) in zip(out, cat_output_pt):
                    self.assertTrue(
                        torch.allclose(out_ait, out_pt, atol=5e-2, rtol=5e-2)
                    )

    def test_skip_parallel_gemm_cat_groups(self):
        self._skip_fuse_parallel_gemm_output_cat(
            b=4,
            ms=[256, 512],
            n=128,
            k=64,
            perm102_bmm_op="perm102_bmm_rrr_bias",
        )


filter_test_cases_by_test_env(ParallelGemmCatFusionTestCase)

if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
