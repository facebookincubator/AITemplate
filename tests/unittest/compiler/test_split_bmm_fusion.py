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
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    has_op,
)
from aitemplate.utils import graph_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SplitBmmFusionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

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
        dtype="float16",
    ):
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

        a = get_random_torch_tensor([B, M, K], dtype)
        b = get_random_torch_tensor([B, N, K], dtype)
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
        module = compile_model(Y, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        if not with_padding:
            assert len(graph) == 3, (
                f"The final graph should have only 3 tensors. "
                f"But it has {len(graph)} tensors now."
            )
        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors({"input0": a, "input1": b}, [y])
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
        dtype="float16",
    ):
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
        module = compile_model(Y, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        assert len(graph) == 3, (
            f"The final graph should have only 3 tensors. "
            f"But it has {len(graph)} tensors now."
        )

        for M in Ms:
            a = get_random_torch_tensor([B, M, K], dtype)
            b = get_random_torch_tensor([B, N, K], dtype)
            xs = a.split(split_size_or_sections, split_dim)
            ys = b.split(split_size_or_sections, split_dim)
            cs = []
            for i in range(n):
                x = xs[i]
                y = ys[i]
                c = torch.bmm(x, y.permute(0, 2, 1))
                cs.append(c)
            y_pt = torch.cat(cs, dim=split_dim)

            y = get_torch_empty_tensor(y_pt.size(), dtype)
            module.run_with_tensors({"input0": a, "input1": b}, [y])
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

    def _test_split_bmm_rcr_fusion_qkv(
        self,
        B,  # batch_size * num_heads * 3
        M,
        N,
        K,
        NH,  # num_heads
        split_size_or_sections,
        split_dim=0,
        testname="test_split_qkv",
        dtype="float16",
        should_fail=False,
    ):
        X = Tensor(
            shape=[B, M, K],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        scale = Tensor(shape=[], dtype=dtype, name="scale", value=K**-0.5)

        (Q, KK, V) = ops.split()(X, split_size_or_sections, split_dim)
        QK = ops.bmm_rcr()(Q, KK)
        QK = ops.elementwise(FuncEnum.MUL)(QK, scale)
        QK = ops.softmax()(QK, -1)
        Y = ops.bmm_rrr_permute((NH,))(QK, V)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        a = get_random_torch_tensor([B, M, K], dtype)
        (q, k, v) = a.split(split_size_or_sections, split_dim)
        qk = torch.bmm(q, k.permute(0, 2, 1)) * K**-0.5
        qk = torch.softmax(qk, -1)
        qkv = torch.bmm(qk, v)
        y_pt = qkv.reshape(B // 3 // NH, NH, M, K).permute([0, 2, 1, 3])

        # Gen module.
        target = detect_target()
        module = compile_model(Y, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(graph)
        if should_fail:
            assert has_op(sorted_ops, "split"), "The final graph should have split op!"
        else:
            assert not has_op(sorted_ops, "split"), "The final graph has split op!"

        y = get_torch_empty_tensor(y_pt.size(), dtype)
        module.run_with_tensors({"input0": a}, [y])
        torch.testing.assert_close(y, y_pt, atol=1e-2, rtol=1e-2)

    def test_split_bmm_rcr_fusion_qkv_sm80(self):
        self._test_split_bmm_rcr_fusion_qkv(3, 4096, 4096, 512, 1, 1)
        self._test_split_bmm_rcr_fusion_qkv(3 * 16, 1024, 1024, 256, 16, 16)

    def test_split_bmm_fusion_fp32_sm80(self):
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
            dtype="float",
        )
        # bmm_rcr_n1, split_dim = 2
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr_n1,
            1,
            10000,
            1,
            5,
            [2, 3],
            2,
            "test_split_bmm_rcr_float",
            dtype="float",
        )
        # bmm_rcr
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr,
            10,
            8,
            32,
            16 * 2,
            16,
            2,
            "test_split_bmm_rcr_float",
            dtype="float",
        )
        # bmm_rcr, split_dim = 0, can only be static
        self._test_split_bmm_rcr_fusion(
            ops.bmm_rcr,
            10,
            8,
            32,
            16 * 2,
            32,
            0,
            "test_split_bmm_rcr_float",
            dtype="float",
        )
        # bmm_rcr_n1
        self._test_split_bmm_rcr_fusion_dynamic_M(
            ops.bmm_rcr_n1,
            1,
            [100, 160],
            1,
            32,
            8,
            2,
            "test_split_bmm_rcr_n1_dynamic_M_float",
            dtype="float",
        )
        # bmm_rcr
        self._test_split_bmm_rcr_fusion_dynamic_M(
            ops.bmm_rcr,
            10,
            [8, 16],
            32,
            16 * 2,
            16,
            2,
            "test_split_bmm_rcr_dynamic_M_float",
            dtype="float",
        )
        self._test_split_bmm_rcr_fusion_qkv(3 * 16, 10, 10, 8, 16, 16, dtype="float")


filter_test_cases_by_test_env(SplitBmmFusionTestCase)

if __name__ == "__main__":
    unittest.main()
