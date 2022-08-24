# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BMMAlphaTestCase(unittest.TestCase):
    def _test_bmm_alpha(
        self,
        bmm_op,
        is_div,
        X_trans,
        W_trans,
        B,
        M,
        N,
        K,
        cst_val,
        expected_num_tensors,
        expected_num_ops,
        use_fp16_acc=False,
        with_add=False,
    ):
        target = detect_target(use_fp16_acc=use_fp16_acc)
        if X_trans:
            X = Tensor(shape=[B, K, M], dtype="float16", name="input_0", is_input=True)
        else:
            X = Tensor(shape=[B, M, K], dtype="float16", name="input_0", is_input=True)
        if W_trans:
            W = Tensor(shape=[B, N, K], dtype="float16", name="input_1", is_input=True)
        else:
            W = Tensor(shape=[B, K, N], dtype="float16", name="input_1", is_input=True)
        if with_add:
            D = Tensor(shape=[B, M, N], dtype="float16", name="input_2", is_input=True)
        BMM_OP = bmm_op()
        Y1 = BMM_OP(X, W, D) if with_add else BMM_OP(X, W)
        elem_func_type = FuncEnum.DIV if is_div else FuncEnum.MUL
        Y2 = ops.elementwise(elem_func_type)(Y1, Tensor([], value=cst_val))
        Y2._attrs["name"] = "output_0"
        Y2._attrs["is_output"] = True
        module = gen_execution_module(
            Y2, target, "./tmp", f"bmm_alpha_{B}_{M}_{N}_{K}_{use_fp16_acc}"
        )
        expected_cst_val = 1.0 / float(cst_val) if is_div else float(cst_val)

        bmm_tensor = None
        bmm_op = None
        for tensor in module.debug_sorted_graph:
            if len(tensor.src_ops()) != 1:
                continue
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                self.assertIsNone(bmm_tensor, "multiple bmm tensor found")
                bmm_tensor = tensor
                bmm_op = src_op
        self.assertIsNotNone(bmm_tensor, "No bmm_tensor found")

        np.testing.assert_allclose(
            bmm_op._attrs["alpha"], 1.0 / float(expected_cst_val), atol=1e2, rtol=1e2
        )
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)
        if X_trans:
            X_pt = torch.randn(B, K, M).cuda().half()
        else:
            X_pt = torch.randn(B, M, K).cuda().half()
        if W_trans:
            W_pt = torch.randn(B, N, K).cuda().half()
        else:
            W_pt = torch.randn(B, K, N).cuda().half()
        if with_add:
            D_pt = torch.randn(B, M, N).cuda().half()

        def pt_bmm():
            XT = torch.transpose(X_pt, 2, 1) if X_trans else X_pt
            WT = torch.transpose(W_pt, 2, 1) if W_trans else W_pt
            Y_pt_1 = torch.bmm(XT, WT)
            Y_pt_2 = Y_pt_1 * expected_cst_val
            if with_add:
                Y_pt_2 = Y_pt_2 + D_pt
            return Y_pt_2

        Y_pt = pt_bmm()

        inputs = {"input_0": X_pt, "input_1": W_pt}
        if with_add:
            inputs["input_2"] = D_pt
        y = torch.empty([B, M, N]).cuda().half()
        module.RunWithTensors(inputs, [y])

        if X_pt.nelement() == 0 or W_pt.nelement() == 0:
            pass
        else:
            self.assertTrue(torch.allclose(Y_pt, y, atol=0.1, rtol=0.1))

    def test_bmm_alpha(self):
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr,
            is_div=True,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=3,
            K=0,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr,
            is_div=True,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=0,
            K=32,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr,
            is_div=True,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=3,
            K=32,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr,
            is_div=False,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=3,
            K=32,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=False,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rrr,
            is_div=False,
            X_trans=False,
            W_trans=False,
            B=2,
            M=2,
            N=3,
            K=8,
            # Padding on N is applied.
            expected_num_tensors=4,
            expected_num_ops=2,
            cst_val=4.32,
            use_fp16_acc=False,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_ccr,
            is_div=True,
            X_trans=True,
            W_trans=True,
            B=2,
            M=11,
            N=8,
            K=3,
            # Padding on M is applied.
            expected_num_tensors=7,
            expected_num_ops=4,
            cst_val=0.32,
            use_fp16_acc=False,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_crr,
            is_div=True,
            X_trans=True,
            W_trans=False,
            B=2,
            M=11,
            N=8,
            K=3,
            # Padding on M is applied.
            expected_num_tensors=6,
            expected_num_ops=3,
            cst_val=0.32,
            use_fp16_acc=False,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rrr_add,
            is_div=False,
            X_trans=False,
            W_trans=False,
            B=2,
            M=12,
            N=8,
            K=4,
            cst_val=0.32,
            expected_num_tensors=4,
            expected_num_ops=1,
            use_fp16_acc=True,
            with_add=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_ccr_add,
            is_div=True,
            X_trans=True,
            W_trans=True,
            B=2,
            M=12,
            N=8,
            K=4,
            cst_val=0.32,
            expected_num_tensors=4,
            expected_num_ops=1,
            use_fp16_acc=True,
            with_add=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr_n1,
            is_div=True,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=1,
            K=32,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=True,
        )
        self._test_bmm_alpha(
            bmm_op=ops.bmm_rcr_n1,
            is_div=False,
            X_trans=False,
            W_trans=True,
            B=1,
            M=1000000,
            N=1,
            K=32,
            expected_num_tensors=3,
            expected_num_ops=1,
            cst_val=2.3,
            use_fp16_acc=False,
        )


if __name__ == "__main__":
    unittest.main()
