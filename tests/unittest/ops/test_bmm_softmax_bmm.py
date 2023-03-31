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
# BMM + Softmax + BMM
# (B, M, K) * (B, N, K) = (B, M, N) #RCR
# softmax on dim N (B, M, N)
# (B, M, N) * (B, N, O) = (B, M, O) #RRR
import itertools
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import filter_test_cases_by_test_env
from aitemplate.utils import shape_utils


def build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


class BMMSoftmaxBMMTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BMMSoftmaxBMMTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_bmm_permute(
        self,
        bs,
        ms,
        N,
        K,
        D,
        head_dim=64,
        num_heads=12,
        causal=False,
        test_name="ck_attn",
        copy_op=False,
    ):
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

        OP = ops.bmm_softmax_bmm_permute(shape=(num_heads,), scale=scale, causal=causal)
        if copy_op:
            OP = ops.bmm_softmax_bmm_permute(**OP._get_op_attributes())
        Y = OP(X, B0, B1)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", f"bmm_{test_name}_permute", dll_name=dll_name
        )

        for b, m in itertools.product(bs, ms):
            X_pt = torch.randn(b, m, K).cuda().half()  # Q
            W_pt = torch.randn(b, N, K).cuda().half()  # K
            B1_pt = torch.randn(b, N, D).cuda().half()  # V

            attn = (X_pt @ W_pt.transpose(-2, -1)) * scale

            if causal:
                bsz = 1
                tgt_len = m
                src_len = N
                causal_attention_mask = build_causal_attention_mask(
                    bsz, m, attn.dtype
                ).to(attn.device)
                attn_weights = attn.reshape(bsz, num_heads, m, N)
                if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                        f" {causal_attention_mask.size()}"
                    )
                attn_weights = (
                    attn_weights.view(bsz, num_heads, tgt_len, src_len)
                    + causal_attention_mask
                )
                attn = attn_weights.view(bsz * num_heads, tgt_len, src_len)

            attn = attn.softmax(dim=-1)
            Y_l = attn @ B1_pt
            Y_r = Y_l.reshape(b // num_heads, num_heads, m, D)
            Y2_pt = torch.permute(Y_r, [0, 2, 1, 3])

            y = torch.empty([b // num_heads, m, num_heads, D]).cuda().half()
            module.run_with_tensors([X_pt, W_pt, B1_pt], [y])
            if X_pt.nelement() == 0 or Y2_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y2_pt, y, atol=1e-1, rtol=1e-1))

            # benchmark
            # time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
            #     [X_pt, W_pt, B1_pt], [y], count=200, repeat=2
            # )

    def _test_b2b(
        self, bs, ms, N, K, D, head_dim=64, test_name="ck_attn", copy_op=False
    ):
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

        OP = ops.bmm_softmax_bmm(scale=scale)
        if copy_op:
            OP = ops.bmm_softmax_bmm(OP._get_op_attributes())
        Y = OP(X, B0, B1)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", f"bmm_{test_name}_permute", dll_name=dll_name
        )

        for b, m in itertools.product(bs, ms):
            X_pt = torch.randn(b, m, K).cuda().half()  # Q
            W_pt = torch.randn(b, N, K).cuda().half()  # K
            B1_pt = torch.randn(b, N, D).cuda().half()  # V

            attn = (X_pt @ W_pt.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            Y2_pt = attn @ B1_pt

            y = torch.empty([b, m, D]).cuda().half()
            module.run_with_tensors([X_pt, W_pt, B1_pt], [y])
            if X_pt.nelement() == 0 or Y2_pt.nelement() == 0:
                pass
            else:
                self.assertTrue(torch.allclose(Y2_pt, y, atol=1e-1, rtol=1e-1))

    def test_rcr_rocm(self):
        # FIXME: re-enable it after we fix the missing parameter for bmm_softmax_bmm
        # self._test_b2b([16], [576], N=576, K=64, D=64, test_name="static")
        self._test_bmm_permute([24], [256], N=256, K=64, D=64, test_name="static")
        self._test_bmm_permute([24], [196], N=196, K=64, D=64, test_name="static")
        self._test_bmm_permute([24], [128], N=1024, K=64, D=128, test_name="static")
        self._test_bmm_permute([24], [128], N=49, K=64, D=128, test_name="static")
        self._test_bmm_permute([24], [49], N=49, K=64, D=64, test_name="static")
        self._test_bmm_permute([24], [1020], N=1020, K=64, D=128, test_name="static")
        self._test_bmm_permute(
            [24], [1020], N=1020, K=64, D=128, test_name="static_copy_op", copy_op=True
        )
        self._test_bmm_permute(
            [32], [49], N=49, K=64, D=64, num_heads=4, test_name="static"
        )
        self._test_bmm_permute(
            [16], [4096], N=64, K=40, D=40, num_heads=8, test_name="static"
        )
        self._test_bmm_permute(
            [12], [64], N=64, K=64, D=64, num_heads=12, causal=True, test_name="static"
        )
        self._test_bmm_permute(
            [12],
            [64],
            N=64,
            K=64,
            D=64,
            num_heads=12,
            causal=True,
            test_name="static_copy_op",
            copy_op=True,
        )


filter_test_cases_by_test_env(BMMSoftmaxBMMTestCase)


if __name__ == "__main__":
    unittest.main()
