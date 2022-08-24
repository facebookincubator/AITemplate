# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# BMM + Softmax + BMM
# (B, M, K) * (B, N, K) = (B, M, N) #RCR
# softmax on dim N (B, M, N)
# (B, M, N) * (B, N, O) = (B, M, O) #RRR
import itertools
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class GEMMTestCase(unittest.TestCase):
    def _test_bmm_permute(self, bs, ms, N, K, D, head_dim=64, test_name="ck_attn"):
        return True  # enable it when ck is ready
        target = detect_target()
        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        m_dim = shape_utils.gen_int_var_min_max(ms, name="m")
        X = Tensor(
            shape=[batch_dim, m_dim, K], dtype="float16", name="input_0", is_input=True
        )
        B0 = Tensor(
            shape=[batch_dim, N, K], dtype="float16", name="input_1", is_input=True
        )
        B1 = Tensor(
            shape=[batch_dim, N, D], dtype="float16", name="input_2", is_input=True
        )

        scale = head_dim**-0.5
        num_heads = 12

        OP = ops.bmm_softmax_bmm_permute(shape=(num_heads,), scale=scale)
        Y = OP(X, B0, B1)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = gen_execution_module(
            Y, target, "./tmp", "bmm_{}_permute".format(test_name)
        )

        for (b, m) in itertools.product(bs, ms):
            X_pt = torch.randn(b, m, K).cuda().half()  # Q
            W_pt = torch.randn(b, N, K).cuda().half()  # K
            B1_pt = torch.randn(b, N, D).cuda().half()  # V

            attn = (X_pt @ W_pt.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            Y_l = attn @ B1_pt
            Y_r = Y_l.reshape(b // num_heads, num_heads, m, D)
            Y2_pt = torch.permute(Y_r, [0, 2, 1, 3])

            y = torch.empty([b // num_heads, m, num_heads, D]).cuda().half()
            module.RunWithTensors([X_pt, W_pt, B1_pt], [y])
            if X_pt.nelement() == 0 or Y2_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y2_pt, y, atol=1e-1, rtol=1e-1))

            # benchmark
            # time_per_iter_ms, time_std, _ = module.BenchmarkWithTensors(
            #     [X_pt, W_pt, B1_pt], [y], count=200, repeat=2
            # )

    def test_rcr(self):
        self._test_bmm_permute([24], [128], N=1024, K=64, D=128, test_name="static")


if __name__ == "__main__":
    unittest.main()
