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
Unittests for flash_attention Operator.
"""
import logging
import math
import os
import unittest

import torch
import torch.nn.functional as F

from aitemplate.compiler import compile_model, Model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import benchmark_pt, detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype

from einops import rearrange, repeat


_LOGGER = logging.getLogger(__name__)


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
    Y_pt = torch.permute(y_pt, [0, 2, 1, 3])  # [1,4096,12,64]
    return Y_pt


def ref_cross_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0):
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, attn_bias=attn_bias)
    q = q.float()
    k = k.float()
    v = v.float()

    q = q * (1 / q.shape[-1] ** 0.5)
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias):
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    out = ref_cross_attention(T(q), T(k), T(v), attn_bias)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


class attentionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _test_flash_attention(
        self,
        batch_size=16,
        nheads=16,
        seqlen=1024,
        n=1024,
        dropout_p=0.0,
        causal=False,
        dtype="float16",
        device="cuda",
        test_name="flash_attention",
        rebuild=True,
        benchmark_pt=False,
        copy_op=False,
    ):
        torch_dtype = string_to_torch_dtype(dtype)
        d = n // nheads

        x = torch.randn(
            batch_size,
            seqlen,
            n,
            device="cuda",
            dtype=torch_dtype,
            requires_grad=True,
        )
        Wqkv = torch.nn.Linear(
            nheads * d,
            3 * nheads * d,
            device=device,
            dtype=torch_dtype,
        )

        lengths = torch.tensor(
            [seqlen] * batch_size, dtype=torch.int, device="cuda"
        ).reshape(-1, 1)
        attention_mask_bool = (
            repeat(torch.arange(seqlen, device="cuda"), "s -> b s", b=batch_size)
            < lengths
        )
        attention_mask = torch.zeros(
            batch_size,
            seqlen,
            device="cuda",
            dtype=torch_dtype,
        )
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
            dtype=dtype,
            name="qkv",
            is_input=True,
        )
        X2 = Tensor(
            shape=[batch_size + 1],
            dtype="int32",
            name="cu_seqlens",
            is_input=True,
        )

        flash_attention_op = ops.flash_attention(
            batch_size=batch_size,
            dropout=dropout_p,
            max_seq_len=max_seqlen_in_batch,
            causal=causal,
        )
        if copy_op:
            flash_attention_op = ops.flash_attention(
                **flash_attention_op._get_op_attributes()
            )
        Y = flash_attention_op(X1, X2)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        x1 = qkv_unpad.detach().to(torch_dtype).cuda()
        x2 = cu_seqlens.detach().to(torch.int32).cuda()
        inputs = {"qkv": x1, "cu_seqlens": x2}
        y = torch.empty(
            [total, num_heads, head_size],
            dtype=torch_dtype,
            device="cuda",
        )
        module.run_with_tensors(inputs, [y])

        # Warm up.
        for _ in range(5):
            module.run_with_tensors(inputs, [y])
        # Benchmark.
        time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
            inputs,
            [y],
            count=100,
        )
        _LOGGER.info(f"benchmark flash-attn time: {time_per_iter_ms}")

        y = y.reshape((batch_size, -1, nheads, d))
        torch.testing.assert_close(y, y_pt, atol=1e-3, rtol=1e-3)

        if benchmark_pt:
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = attention_ref
            args = (
                qkv.to(torch_dtype).cuda(),
                attention_mask_bool.cuda(),
                dropout_p,
                False,
                False,
            )
            duration = benchmark_torch_function(100, func, *args)
            print(
                f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_flash_attention(self):
        self._test_flash_attention(
            test_name="flash_attention_fp16",
            dtype="float16",
        )
        self._test_flash_attention(
            test_name="flash_attention_fp16_copy_op",
            copy_op=True,
            dtype="float16",
        )

    def _test_attention(
        self,
        test_name="attention",
        rebuild=True,
        benchmark=False,
        dtype="float16",
    ):
        target = detect_target()
        nheads = 12
        d = 64  # head_dim
        seqlen = 4096
        dim = 768
        token_emb_init_range = 0.001
        X = Tensor(
            shape=[seqlen, dim],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        qkv_w = Tensor(
            shape=[dim * 3, dim],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[dim * 3],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )

        qkv = ops.gemm_rcr_bias_permute(shape=(seqlen, 3, nheads), layout="m2n3")(
            X, qkv_w, B
        )
        (q, k, v) = ops.split()(qkv, 1, dim=0)
        scale = Tensor(
            shape=[],
            dtype=dtype,
            name="input_3",
            value=(d**-0.5),
        )
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

        X_pt = get_random_torch_tensor([seqlen, dim], dtype=dtype)
        X_pt *= token_emb_init_range
        W_pt = get_random_torch_tensor([dim * 3, dim], dtype=dtype)
        B_pt = get_random_torch_tensor([dim * 3], dtype=dtype)
        Y_pt = attention_pt(X_pt, W_pt, B_pt, nheads, d, seqlen)
        inputs = {
            "input_0": X_pt,
            "input_1": W_pt,
            "input_2": B_pt,
        }
        torch_dtype = string_to_torch_dtype(dtype)
        y = torch.empty_like(Y_pt, dtype=torch_dtype)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(y, Y_pt, atol=1e-1, rtol=1e-1)

        if benchmark:
            pt_time = benchmark_pt.benchmark_torch_function(
                100, attention_pt, X_pt, W_pt, B_pt, nheads, d, seqlen
            )
            _LOGGER.info(f"benchmark compiler model time: {pt_time}")

            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark.
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            _LOGGER.info(f"benchmark compiler model time: {time_per_iter_ms}")

    @unittest.skipIf(detect_target().name() == "cuda", "Not supported by CUDA.")
    def test_rocm_attention(self):
        self._test_attention(
            test_name="attention_fp16",
            dtype="float16",
        )

    def _test_mem_eff_attention(
        self,
        batch_size=16,
        nheads=16,
        seqlen=1024,
        n=1024,
        dropout_p=0.0,
        causal=False,
        dtype="float16",
        device="cuda",
        test_name="mem_eff_attention",
        rebuild=True,
        benchmark_ait=False,
        benchmark_pt=False,
        copy_op=False,
        use_perm=True,
        atol=1e-3,
        rtol=1e-3,
    ):
        torch_dtype = string_to_torch_dtype(dtype)
        d = n // nheads

        x = torch.randn(
            batch_size,
            seqlen,
            n,
            device="cuda",
            dtype=torch_dtype,
            requires_grad=True,
        )
        Wqkv = torch.nn.Linear(
            nheads * d,
            3 * nheads * d,
            device=device,
            dtype=torch_dtype,
        )

        lengths = torch.tensor(
            [seqlen] * batch_size, dtype=torch.int, device="cuda"
        ).reshape(-1, 1)
        attention_mask_bool = (
            repeat(torch.arange(seqlen, device="cuda"), "s -> b s", b=batch_size)
            < lengths
        )
        attention_mask = torch.zeros(
            batch_size,
            seqlen,
            device="cuda",
            dtype=torch_dtype,
        )
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
        q, k, v = torch.split(qkv, 1, dim=2)
        output = attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)
        y_pt = output.detach()

        total, _, num_heads, head_size = qkv_unpad.shape

        Q = Tensor(
            shape=[batch_size, num_heads, seqlen, head_size],
            dtype=dtype,
            name="q",
            is_input=True,
        )
        K = Tensor(
            shape=[batch_size, num_heads, seqlen, head_size],
            dtype=dtype,
            name="k",
            is_input=True,
        )
        V = Tensor(
            shape=[batch_size, num_heads, seqlen, head_size],
            dtype=dtype,
            name="v",
            is_input=True,
        )

        mem_eff_attention_op = ops.mem_eff_attention(
            causal=causal,
        )
        if copy_op:
            mem_eff_attention_op = ops.mem_eff_attention(
                **mem_eff_attention_op._get_op_attributes()
            )

        Y = mem_eff_attention_op(Q, K, V)

        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        q = torch.permute(q, (0, 3, 2, 1, 4)).reshape(
            batch_size, num_heads, seqlen, head_size
        )
        k = torch.permute(k, (0, 3, 2, 1, 4)).reshape(
            batch_size, num_heads, seqlen, head_size
        )
        v = torch.permute(v, (0, 3, 2, 1, 4)).reshape(
            batch_size, num_heads, seqlen, head_size
        )

        inputs = {
            "q": q.detach().to(torch_dtype).cuda().contiguous(),
            "k": k.detach().to(torch_dtype).cuda().contiguous(),
            "v": v.detach().to(torch_dtype).cuda().contiguous(),
        }

        y = torch.empty(
            [batch_size, seqlen, num_heads, head_size],
            dtype=torch_dtype,
            device="cuda",
        )
        module.run_with_tensors(inputs, [y])

        if benchmark_ait:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark AIT
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            _LOGGER.info(f"benchmark eff-mem-attn time: {time_per_iter_ms}")

        torch.testing.assert_close(y, y_pt.to(torch_dtype), atol=atol, rtol=rtol)

        if benchmark_pt:
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = attention_ref
            args = (
                qkv.to(torch_dtype).cuda(),
                attention_mask_bool.cuda(),
                dropout_p,
                False,
                False,
            )
            duration = benchmark_torch_function(100, func, *args)
            print(
                f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_mem_eff_attention_fp16(self):
        for use_perm in [False, True]:
            self._test_mem_eff_attention(
                use_perm=use_perm,
                test_name=f"mem_eff_attention_fp16_{use_perm}",
                dtype="float16",
            )
            self._test_mem_eff_attention(
                use_perm=use_perm,
                causal=True,
                test_name=f"mem_eff_attention_fp16_{use_perm}_causal",
                dtype="float16",
            )
            self._test_mem_eff_attention(
                batch_size=16,
                nheads=4,
                seqlen=8,
                n=80,
                use_perm=use_perm,
                test_name="mem_eff_attention_fp16_nheads_20",
                dtype="float16",
            )
            # self._test_mem_eff_attention(batch_size=1, nheads=8, seqlen=8, n=64, use_perm=use_perm, test_name="mem_eff_attention1")
            # self._test_mem_eff_attention(batch_size=16, nheads=8, seqlen=8, n=512, use_perm=use_perm, test_name="mem_eff_attention2")
            # self._test_mem_eff_attention(batch_size=16, nheads=8, seqlen=8, n=1024, use_perm=use_perm, test_name="mem_eff_attention3")
            # self._test_mem_eff_attention(batch_size=16, nheads=8, seqlen=16, n=1024, use_perm=use_perm, test_name="mem_eff_attention4")
            # self._test_mem_eff_attention(batch_size=1, nheads=8, seqlen=16, n=64, use_perm=use_perm, test_name="mem_eff_attention5")

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.expectedFailure
    def test_mem_eff_attention_invalid_head_size_fp16(self):
        self._test_mem_eff_attention(
            batch_size=16,
            nheads=8,
            seqlen=8,
            n=80,
            test_name="mem_eff_attention_fp16_invalid_head_size",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_mem_eff_attention_fp32(self):
        for use_perm in [False, True]:
            self._test_mem_eff_attention(
                use_perm=use_perm,
                test_name=f"mem_eff_attention_fp32_{use_perm}",
                dtype="float32",
            )
            self._test_mem_eff_attention(
                use_perm=use_perm,
                causal=True,
                test_name=f"mem_eff_attention_fp32_{use_perm}_causal",
                dtype="float32",
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_mem_eff_attention_bf16(self):
        for use_perm in [False, True]:
            self._test_mem_eff_attention(
                use_perm=use_perm,
                test_name=f"mem_eff_attention_bf16_{use_perm}",
                dtype="bfloat16",
                atol=1e-2,
                rtol=1e-2,
            )
            self._test_mem_eff_attention(
                use_perm=use_perm,
                causal=True,
                test_name=f"mem_eff_attention_bf16_{use_perm}_causal",
                dtype="bfloat16",
                atol=1e-2,
                rtol=1e-2,
            )

    def _test_cross_attention(
        self,
        batch_size=16,
        num_heads=16,
        seqlen=1024,
        seqlen_kv=1024,
        head_size=64,
        head_size_v=64,
        dropout_p=0.0,
        causal=False,
        dtype="float16",
        device="cuda",
        test_name="cross_attention",
        rebuild=True,
        benchmark_ait=False,
        benchmark_pt=False,
        copy_op=False,
        atol=1e-3,
        rtol=1e-3,
    ):
        torch_dtype = string_to_torch_dtype(dtype)

        q = torch.randn(
            batch_size,
            seqlen,
            num_heads,
            head_size,
            device="cuda",
            dtype=torch_dtype,
        )
        k = torch.randn(
            batch_size,
            seqlen_kv,
            num_heads,
            head_size,
            device="cuda",
            dtype=torch_dtype,
        )
        v = torch.randn(
            batch_size,
            seqlen_kv,
            num_heads,
            head_size_v,
            device="cuda",
            dtype=torch_dtype,
        )

        output = ref_cross_attention(q, k, v)
        y_pt = output.detach()

        Q = Tensor(
            shape=[batch_size, num_heads, seqlen, head_size],
            dtype=dtype,
            name="q",
            is_input=True,
        )
        K = Tensor(
            shape=[batch_size, num_heads, seqlen_kv, head_size],
            dtype=dtype,
            name="k",
            is_input=True,
        )
        V = Tensor(
            shape=[batch_size, num_heads, seqlen_kv, head_size_v],
            dtype=dtype,
            name="v",
            is_input=True,
        )

        mem_eff_attention_op = ops.mem_eff_attention(
            causal=causal,
        )
        if copy_op:
            mem_eff_attention_op = ops.mem_eff_attention(
                **mem_eff_attention_op._get_op_attributes()
            )
        Y = mem_eff_attention_op(Q, K, V)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        q = torch.permute(q, (0, 2, 1, 3))
        k = torch.permute(k, (0, 2, 1, 3))
        v = torch.permute(v, (0, 2, 1, 3))

        inputs = {
            "q": q.detach().to(torch_dtype).cuda().contiguous(),
            "k": k.detach().to(torch_dtype).cuda().contiguous(),
            "v": v.detach().to(torch_dtype).cuda().contiguous(),
        }
        y = torch.empty(
            [batch_size, seqlen, num_heads, head_size_v],
            dtype=torch_dtype,
            device="cuda",
        )
        module.run_with_tensors(inputs, [y])

        if benchmark_ait:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark AIT
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            _LOGGER.info(f"benchmark cross-attn time: {time_per_iter_ms}")

        torch.testing.assert_close(y, y_pt.to(torch_dtype), atol=atol, rtol=rtol)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_cross_attention_fp16(self):
        self._test_cross_attention(
            test_name="cross_attention_fp16",
            dtype="float16",
        )
        self._test_cross_attention(
            seqlen=1024,
            seqlen_kv=768,
            head_size=64,
            head_size_v=64,
            test_name="cross_attention2_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_cross_attention_fp32(self):
        self._test_cross_attention(
            test_name="cross_attention_fp32",
            dtype="float32",
        )
        self._test_cross_attention(
            seqlen=1024,
            seqlen_kv=768,
            head_size=64,
            head_size_v=64,
            test_name="cross_attention2_fp32",
            dtype="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by CUDA < SM80.",
    )
    def test_cross_attention_bf16(self):
        self._test_cross_attention(
            test_name="cross_attention_bf16",
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        self._test_cross_attention(
            seqlen=1024,
            seqlen_kv=768,
            head_size=64,
            head_size_v=64,
            test_name="cross_attention2_bf16",
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
