# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class FusePermuteBmmRRRN1Case(unittest.TestCase):
    def _test_permute_bmm_rrr_n1(self, B, M, K, testname):
        N = 1

        batch_dim = shape_utils.gen_int_var_min_max(B)
        X = Tensor(
            shape=[batch_dim, M, K], dtype="float16", name="input_0", is_input=True
        )
        W = Tensor(
            shape=[batch_dim, N, K], dtype="float16", name="input_1", is_input=True
        )

        WT = ops.permute021()(W)

        Y = ops.bmm_rrr()(X, WT)
        Y._attrs["name"] = "bmm_rrr_tensor"

        output = ops.elementwise(FuncEnum.COS)(Y)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(output, target, "./tmp", testname)

        bmm_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_rrr_tensor":
                bmm_tensor = tensor
                break

        assert len(bmm_tensor._attrs["src_ops"]) == 1
        src_op = list(bmm_tensor._attrs["src_ops"])[0]
        assert src_op._attrs["op"] == "bmm_rcr_n1"

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()

            Y_pt = torch.cos(torch.bmm(X_pt, W_pt))
            w = W_pt.permute([0, 2, 1]).contiguous()

            # We currently only have row-major outputs.
            y = torch.empty([b, M, N]).cuda().half()
            module.RunWithTensors({"input_0": X_pt, "input_1": w}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_rrr_n1(self):
        self._test_permute_bmm_rrr_n1([1], 4, 8, "permute_bmm_rrr_n1")
        self._test_permute_bmm_rrr_n1([1, 3], 4, 8, "permute_bmm_rrr_n1_dynamic")


if __name__ == "__main__":
    unittest.main()
