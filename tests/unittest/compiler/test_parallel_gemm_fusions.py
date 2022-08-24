# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

from typing import Sequence

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.fuse_parallel_gemms import fuse_parallel_gemms
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils


def _has_op(sorted_ops, op_name):
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type == op_name:
            return True
    return False


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ParallelGemmCatFusionTestCase(unittest.TestCase):
    def _fuse_2_split_parallel_gemm_cat(
        self, b: int, ms: Sequence[int], n: int, k: int
    ):
        logging.info(
            f"_fuse_2_split_parallel_gemm_cat, b: {b}, ms: {ms}, n: {n}, k: {k}"
        )
        X1 = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype="float16",
            name="X1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype="float16",
            name="X2",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(2 * b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype="float16",
                name=f"W{i}",
                is_input=True,
            )
            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype="float16",
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
        assert not _has_op(
            sorted_ops, "perm102_bmm_rrr_bias"
        ), "the final graph should not have op perm102_bmm_rrr_bias"
        assert not _has_op(
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
    ):
        logging.info(f"_fuse_parallel_gemm_cat, b: {b}, ms: {ms}, n: {n}, k: {k}")
        X = Tensor(
            shape=[IntVar(ms, "input_batch"), IntImm(b * k)],
            dtype="float16",
            name="X",
            is_input=True,
        )
        Ws = []
        Bs = []
        for i in range(b):
            W = Tensor(
                shape=[IntImm(n), IntImm(k)],
                dtype="float16",
                name=f"W{i}",
            )
            if reshape_weight:
                W = ops.reshape()(W, [n, k])  # no-op, for testing
            Ws.append(W)
            B = Tensor(
                shape=[IntImm(n)],
                dtype="float16",
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
            constants[f"W{i}"] = torch.randn(n, k).cuda().half()
            constants[f"B{i}"] = torch.randn(n).cuda().half()

        # Gen module.
        target = detect_target()
        with gen_execution_module(
            [cat_output],
            target,
            "./tmp",
            "_fuse_parallel_gemm_cat",
            constants=constants,
        ) as module:
            # Verify the generated graph.
            sorted_graph = module.debug_sorted_graph
            sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
            assert _has_op(
                sorted_ops, perm102_bmm_op
            ), "the final graph does not have op perm102_bmm_rrr_bias"
            if not has_tanh:
                assert not _has_op(
                    sorted_ops, "split"
                ), "the final graph has split op, but it should not"

            for m in ms:
                x_pt = torch.randn(m, b * k).cuda().half()
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

                out = torch.empty([m, b * n]).cuda().half()
                module.RunWithTensors([x_pt], [out])
                # module.BenchmarkWithTensors([x_pt], [out])

                # Do comparisons.
                self.assertTrue(
                    torch.allclose(out, cat_output_pt, atol=1e-2, rtol=1e-2)
                )

    def test_fuse_parallel_gemm_cat(self):
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


if __name__ == "__main__":
    unittest.main()
