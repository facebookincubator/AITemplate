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
"""
Unittests for flash_attenion Operator.
"""
import math
import os
import unittest

import torch
import torch.nn.functional as F

from aitemplate.compiler import compile_model, Model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import benchmark_pt, detect_target
from aitemplate.utils import logger
from einops import rearrange, repeat


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, dim)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, dim), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        index_first_axis(rearrange(hidden_states, "b s d -> (b s) d"), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def index_first_axis(x, indices):
    return torch.gather(x, 0, repeat(indices, "z -> z d", d=x.shape[1]))


def attention_ref(qkv, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output.to(dtype=qkv.dtype)


def attention_pt(X_pt, W_pt, B_pt, nheads, d, seqlen):
    qkv_pt = torch.nn.functional.linear(
        X_pt, W_pt, bias=B_pt
    )  # [4096*3, 256] *[768, 256]
    qkv_pt = torch.reshape(
        qkv_pt, [1, seqlen, 3, nheads, d]
    )  # [4096*3, 768] -> [1, 4096, 3, 12, 64]
    qkv_pt = torch.permute(qkv_pt, [2, 0, 3, 1, 4])  # [3, 1, 12, 4096, 64]

    q_pt, k_pt, v_pt = torch.split(qkv_pt, 1, dim=0)  # [1, 1, 12, 4096, 64]
    scale_pt = torch.tensor(64**-0.5)
    q_pt = q_pt * (scale_pt)
    # #[12, 4096, 64] * [12, 64, 4096] => [12, 4096, 4096]
    attn_pt = torch.bmm(
        (torch.reshape(q_pt, [nheads, -1, d])),
        (torch.transpose(torch.reshape(k_pt, [nheads, -1, d]), 2, 1)),
    )  # [12,4096,4096]
    attn_pt = torch.softmax(attn_pt, dim=-1)  # [12,4096,4096]
    v_pt = torch.reshape(v_pt, [nheads, -1, d])  # [12, 4096, 64]
    y_pt = torch.bmm(attn_pt, v_pt)  # [12, 4096, 64]
    y_pt = torch.reshape(y_pt, [1, nheads, seqlen, d])
    Y_pt = torch.permute(y_pt, [0, 2, 1, 3]).cuda().half()  # [1,4096,12,64]
    return Y_pt


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class attentionTestCase(unittest.TestCase):
    def _test_flash_attention(
        self,
        batch_size=16,
        nheads=16,
        seqlen=1024,
        n=1024,
        dropout_p=0.0,
        causal=False,
        dtype=torch.float16,
        device="cuda",
        test_name="attention",
        rebuild=True,
        benchmark_pt=False,
    ):

        d = n // nheads

        x = torch.randn(
            batch_size, seqlen, n, device="cuda", dtype=dtype, requires_grad=True
        )
        Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

        lengths = torch.tensor(
            [seqlen] * batch_size, dtype=torch.int, device="cuda"
        ).reshape(-1, 1)
        attention_mask_bool = (
            repeat(torch.arange(seqlen, device="cuda"), "s -> b s", b=batch_size)
            < lengths
        )
        attention_mask = torch.zeros(batch_size, seqlen, device="cuda", dtype=dtype)
        attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")

        x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
            x, attention_mask_bool
        )
        qkv_unpad = (
            rearrange(Wqkv(x_unpad), "nnz (t h d) -> nnz t h d", t=3, h=nheads)
            .detach()
            .requires_grad_()
        )
        qkv = (
            rearrange(Wqkv(x), "b s (t h d) -> b s t h d", t=3, h=nheads)
            .detach()
            .requires_grad_()
        )
        output = attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)
        y_pt = output.detach()

        total, _, num_heads, head_size = qkv_unpad.shape

        X1 = Tensor(
            shape=[total, 3, num_heads, head_size],
            dtype="float16",
            name="qkv",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_size + 1],
            dtype="int32",
            name="cu_seqlens",
            is_input=True,
        )
        Y = ops.flash_attention(
            batch_size=batch_size,
            dropout=dropout_p,
            max_seq_len=max_seqlen_in_batch,
            causal=causal,
        )(X1, X2)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        x1 = qkv_unpad.detach().half().cuda()
        x2 = cu_seqlens.detach().to(torch.int32).cuda()
        inputs = {"qkv": x1, "cu_seqlens": x2}
        y = torch.empty([total, num_heads, head_size]).cuda().half()
        module.run_with_tensors(inputs, [y])
        y = y.reshape((batch_size, -1, nheads, d))

        self.assertTrue(torch.allclose(y_pt, y, atol=1e-3, rtol=1e-3))

        if benchmark_pt:
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = attention_ref
            args = (
                qkv.cuda().half(),
                attention_mask_bool.cuda(),
                dropout_p,
                False,
                False,
            )
            duration = benchmark_torch_function(100, func, *args)
            print(
                f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
            )

    def test_flash_attention(self):
        if detect_target().name() == "cuda":
            self._test_flash_attention(test_name="flash_attention")

    def _test_attention(self, test_name, rebuild=True, benchmark=False):
        target = detect_target()
        nheads = 12
        d = 64  # head_dim
        seqlen = 4096
        dim = 768
        token_emb_init_range = 0.001
        X = Tensor(shape=[seqlen, dim], dtype="float16", name="input_0", is_input=True)
        qkv_w = Tensor(
            shape=[dim * 3, dim], dtype="float16", name="input_1", is_input=True
        )
        B = Tensor(shape=[dim * 3], dtype="float16", name="input_2", is_input=True)

        qkv = ops.gemm_rcr_bias_permute(shape=(seqlen, 3, nheads), layout="m2n3")(
            X, qkv_w, B
        )
        (q, k, v) = ops.split()(qkv, 1, dim=0)
        scale = Tensor(shape=[], dtype="float16", name="input_3", value=(d**-0.5))
        q = ops.elementwise(FuncEnum.MUL)(q, scale)
        attn = ops.bmm_rcr()(
            (ops.reshape()(q, [nheads, -1, d])),
            (ops.reshape()(k, [nheads, -1, d])),
        )
        attn = ops.softmax()(attn, -1)
        v = ops.reshape()(v, [nheads, -1, d])
        Y = ops.bmm_rrr_permute((nheads,))(attn, v)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        X_pt = torch.randn(seqlen, dim).cuda().half() * token_emb_init_range
        W_pt = torch.randn(dim * 3, dim).cuda().half()
        B_pt = torch.randn(dim * 3).cuda().half()
        Y_pt = attention_pt(X_pt, W_pt, B_pt, nheads, d, seqlen)
        inputs = {
            "input_0": X_pt.half(),
            "input_1": W_pt.half(),
            "input_2": B_pt.half(),
        }
        y = torch.empty(Y_pt.shape).cuda().half()
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        if benchmark:
            pt_time = benchmark_pt.benchmark_torch_function(
                100, attention_pt, X_pt, W_pt, B_pt, nheads, d, seqlen
            )
            logger.info(__file__, "benchmark compiler model time: {0}".format(pt_time))

            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark.
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            logger.info(
                __file__, "benchmark compiler model time: {0}".format(time_per_iter_ms)
            )

    def test_attention(self):
        if detect_target().name() == "rocm":
            self._test_attention(test_name="attention")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
