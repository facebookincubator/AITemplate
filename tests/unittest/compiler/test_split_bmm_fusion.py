# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module


class SplitBmmFusionTestCase(unittest.TestCase):
    def _test_split_bmm_rcr_fusion(
        self,
        bmm_rcr_op,
        B,
        M,
        N,
        K,
        split_size_or_sections,
        split_dim,
        testname,
        with_padding=False,
    ):
        dtype = "float16"

        T_A = Tensor(
            shape=[B, M, K],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        T_B = Tensor(
            shape=[B, N, K],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        Xs = ops.split()(T_A, split_size_or_sections, split_dim)
        Ys = ops.split()(T_B, split_size_or_sections, split_dim)
        assert len(Xs) == len(Ys)

        n = len(Xs)
        Cs = []
        for i in range(n):
            X = Xs[i]
            Y = Ys[i]
            C = bmm_rcr_op()(X, Y)
            Cs.append(C)
        Y = ops.concatenate()(Cs, dim=split_dim)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        a = torch.randn(B, M, K).cuda().half()
        b = torch.randn(B, N, K).cuda().half()
        xs = a.split(split_size_or_sections, split_dim)
        ys = b.split(split_size_or_sections, split_dim)
        cs = []
        for i in range(n):
            x = xs[i]
            y = ys[i]
            c = torch.bmm(x, y.permute(0, 2, 1))
            cs.append(c)
        y_pt = torch.cat(cs, dim=split_dim)

        # Gen module.
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        if not with_padding:
            assert len(graph) == 3, (
                f"The final graph should have only 3 tensors. "
                f"But it has {len(graph)} tensors now."
            )
        y = torch.empty(y_pt.size()).cuda().half()
        module.RunWithTensors({"input0": a, "input1": b}, [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    def test_split_bmm_rcr_fusion_static(self):
        # bmm_rcr (K with an odd value) with padding:
        # in this case, split and bmm_rcr are not going to be fused actually because
        # of the padding applied to bmm_rcr.
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr,
            1,
            10000,
            3,
            5,
            [2, 3],
            2,
            "test_split_bmm_rcr",
            with_padding=True,
        )
        # bmm_rcr_n1
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr_n1, 1, 160, 1, 32, 8, 2, "test_split_bmm_rcr"
        )
        # bmm_rcr_n1, split_dim = 2
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr_n1, 1, 10000, 1, 5, [2, 3], 2, "test_split_bmm_rcr"
        )
        # bmm_rcr_n1, split_dim = 2
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr_n1, 1, 10000, 1, 5, [3, 2], 2, "test_split_bmm_rcr"
        )
        # bmm_rcr_n1
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr_n1, 1, 10, 1, 32, [16, 8, 8], 2, "test_split_bmm_rcr"
        )
        # bmm_rcr_n1, split_dim = 0
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 4, 10000, 1, 32, [2, 2], 0, "test_split_bmm_rcr"
        )
        # bmm_rcr_n1, split_dim = 1
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 64, 2, 2, 32, 1, 1, "test_split_bmm_rcr"
        )
        # bmm_rcr
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 1024, 128, 512, 256 * 2, 256, 2, "test_split_bmm_rcr"
        )
        # bmm_rcr
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 1, 10000, 3, 96, [32, 32, 32], 2, "test_split_bmm_rcr"
        )
        # bmm_rcr, split_dim = 0, can only be static
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 1024, 128, 512, 256 * 2, 512, 0, "test_split_bmm_rcr"
        )
        # bmm_rcr, split_dim = 1
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr, 1024, 512, 512, 256 * 2, 256, 1, "test_split_bmm_rcr"
        )

    def _test_split_bmm_rcr_fusion_dynamic_M(
        self,
        bmm_rcr_op,
        B,
        Ms,
        N,
        K,
        split_size_or_sections,
        split_dim,
        testname,
    ):
        dtype = "float16"
        assert isinstance(Ms, (list, tuple))

        T_A = Tensor(
            shape=[B, IntVar(name="input_batch", values=Ms), K],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        T_B = Tensor(
            shape=[B, N, K],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        Xs = ops.split()(T_A, split_size_or_sections, split_dim)
        Ys = ops.split()(T_B, split_size_or_sections, split_dim)
        assert len(Xs) == len(Ys)

        n = len(Xs)
        Cs = []
        for i in range(n):
            X = Xs[i]
            Y = Ys[i]
            C = bmm_rcr_op()(X, Y)
            Cs.append(C)
        Y = ops.concatenate()(Cs, dim=split_dim)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        assert len(graph) == 3, (
            f"The final graph should have only 3 tensors. "
            f"But it has {len(graph)} tensors now."
        )

        for M in Ms:
            a = torch.randn(B, M, K).cuda().half()
            b = torch.randn(B, N, K).cuda().half()
            xs = a.split(split_size_or_sections, split_dim)
            ys = b.split(split_size_or_sections, split_dim)
            cs = []
            for i in range(n):
                x = xs[i]
                y = ys[i]
                c = torch.bmm(x, y.permute(0, 2, 1))
                cs.append(c)
            y_pt = torch.cat(cs, dim=split_dim)

            y = torch.empty(y_pt.size()).cuda().half()
            module.RunWithTensors({"input0": a, "input1": b}, [y])
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    def test_split_bmm_rcr_fusion_dynamic_M(self):
        # bmm_rcr_n1
        self._test_split_bmm_rcr_fusion_dynamic_M(
            ops.bmm_rcr_n1,
            1,
            [100, 160],
            1,
            32,
            8,
            2,
            "test_split_bmm_rcr_n1_dynamic_M",
        )
        # bmm_rcr
        self._test_split_bmm_rcr_fusion_dynamic_M(
            ops.bmm_rcr,
            1024,
            [128, 256],
            512,
            256 * 2,
            256,
            2,
            "test_split_bmm_rcr_dynamic_M",
        )


if __name__ == "__main__":
    unittest.main()
