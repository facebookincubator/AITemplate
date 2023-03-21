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
Unittests for vanilla_attention.
"""
import logging
import math
import os
import unittest

import torch
import torch.nn.functional as F

from aitemplate.compiler import compile_model, Model
from aitemplate.frontend import nn, Tensor
from aitemplate.frontend.nn.vanilla_attention import vanilla_attention
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils
from einops import rearrange


_LOGGER = logging.getLogger(__name__)


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


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
    if attn_mask is not None:
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


class VanillaAttentionTestCase(unittest.TestCase):
    def _test_vanilla_attention(
        self,
        batch_size=16,
        nheads=16,
        seqlen=1024,
        n=1024,
        causal=False,
        dtype=torch.float16,
        device="cuda",
        test_name="attention",
        rebuild=True,
        benchmark_ait=False,
        benchmark_pt=False,
    ):
        head_size = n // nheads

        x = torch.randn(
            batch_size, seqlen, n, device="cuda", dtype=dtype, requires_grad=True
        )
        Wqkv = torch.nn.Linear(
            nheads * head_size, 3 * nheads * head_size, device=device, dtype=dtype
        )
        qkv = (
            rearrange(Wqkv(x), "b s (t h d) -> b s t h d", t=3, h=nheads)
            .detach()
            .requires_grad_()
        )
        q, k, v = torch.split(qkv, 1, dim=2)
        q, k, v = (
            q.squeeze(2),
            k.squeeze(2),
            v.squeeze(2),
        )  # batch_size, seqlen, nheads, head_size
        output = attention_ref(qkv, None, 0, causal=causal)
        y_pt = output.detach()
        y_pt = y_pt.reshape(batch_size, seqlen, nheads * head_size)
        print(f"{y_pt.shape=}")

        Q = Tensor(
            shape=[batch_size, seqlen, nheads, head_size],
            dtype="float16",
            name="q",
            is_input=True,
        )
        K = Tensor(
            shape=[batch_size, seqlen, nheads, head_size],
            dtype="float16",
            name="k",
            is_input=True,
        )
        V = Tensor(
            shape=[batch_size, seqlen, nheads, head_size],
            dtype="float16",
            name="v",
            is_input=True,
        )

        from aitemplate.compiler.base import _TorchConstantTensorData

        causal_mask = None
        if causal:
            mask = torch.triu(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1
            )
            causal_mask_pt = torch.zeros(
                seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
            )
            causal_mask_pt.masked_fill_(mask, float("-inf"))
            causal_mask_pt = causal_mask_pt.unsqueeze(0)

            causal_mask = Tensor(
                shape=[1, seqlen, seqlen],
                dtype="float16",
                name="causal_mask",
            )
            causal_mask._bind_data(_TorchConstantTensorData(causal_mask_pt))
        Y = vanilla_attention(Q, K, V, attn_mask=causal_mask)

        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        if rebuild:
            target = detect_target()
            module = compile_model(Y, target, "./tmp", test_name)
        else:
            module = Model(os.path.join("./tmp", test_name, "test.so"))

        inputs = {
            "q": q.detach().half().cuda().contiguous(),
            "k": k.detach().half().cuda().contiguous(),
            "v": v.detach().half().cuda().contiguous(),
        }

        y = torch.empty([batch_size, seqlen, nheads * head_size]).cuda().half()
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
            _LOGGER.info("benchmark vanilla-attn time: {0}".format(time_per_iter_ms))

        self.assertTrue(torch.allclose(y_pt.half(), y, atol=1e-1, rtol=1e-1))

        if benchmark_pt:
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = attention_ref
            args = (
                qkv.cuda().half(),
                None,
                0,
                False,
                False,
            )
            duration = benchmark_torch_function(100, func, *args)
            print(
                f"PT:  BS: {batch_size}, Time per iter: {duration:.2f}ms, QPS: {batch_size / duration:.2f}"
            )

    def test_vanilla_attention(self):
        self._test_vanilla_attention(test_name="vanilla_attention")
        self._test_vanilla_attention(test_name="vanilla_attention_causal", causal=True)

    def _test_mha(
        self,
        batch_sizes,
        seqlen=1,
        seqlen_kv=62,
        dim=4,
        num_heads=2,
        use_fp16_acc=False,
        benchmark_ait=False,
    ):
        pt_mod = (
            torch.nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
            )
            .cuda()
            .half()
        )
        pt_mod = pt_mod.eval()

        pt_params = dict(pt_mod.named_parameters())
        params_ait = {}
        for key, arr in pt_params.items():
            if "in_proj" in key:
                if len(arr.shape) == 2:
                    w_q, w_k, w_v = arr.chunk(3)
                    params_ait["proj_q_weight"] = w_q
                    params_ait["proj_k_weight"] = w_k
                    params_ait["proj_v_weight"] = w_v
                else:
                    b_q, b_k, b_v = arr.chunk(3)
                    params_ait["proj_q_bias"] = b_q
                    params_ait["proj_k_bias"] = b_k
                    params_ait["proj_v_bias"] = b_v

            else:
                params_ait[key.replace(".", "_").replace("out_proj", "proj")] = arr

        ait_mod = nn.VanillaCrossAttention(
            dim=dim,
            seq_len=seqlen,
            seq_len_kv=seqlen_kv,
            num_heads=num_heads,
            qkv_bias=True,
            has_residual=False,
        )
        ait_mod.name_parameter_tensor()

        if len(batch_sizes) == 1:
            # static
            batch_dim = batch_sizes[0]
        else:
            batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, name="batch_size")

        inputs_ait = Tensor([batch_dim, seqlen, dim], name="input0", is_input=True)
        inputs_ait_k = Tensor([batch_dim, seqlen_kv, dim], name="input1", is_input=True)
        inputs_ait_v = Tensor([batch_dim, seqlen_kv, dim], name="input2", is_input=True)
        Y = ait_mod(inputs_ait, inputs_ait_k, inputs_ait_v)
        Y = Y + inputs_ait
        mark_output(Y)
        target = detect_target(use_fp16_acc=False)
        exe_module = compile_model(Y, target, "./tmp", "cross_attn_dynamic")
        for name, weight in params_ait.items():
            exe_module.set_constant_with_tensor(name, weight)

        for batch_size in batch_sizes:
            input_pt = torch.randn([batch_size, seqlen, dim]).cuda().half()
            if seqlen == seqlen_kv:
                input_pt_k = input_pt
                input_pt_v = input_pt
            else:
                input_pt_k = torch.randn([batch_size, seqlen_kv, dim]).cuda().half()
                input_pt_v = torch.randn([batch_size, seqlen_kv, dim]).cuda().half()

            pt_ys, _ = pt_mod(input_pt, input_pt_k, input_pt_v)
            pt_ys = pt_ys + input_pt
            print("pt output:", pt_ys.shape)

            inputs = {"input0": input_pt, "input1": input_pt_k, "input2": input_pt_v}
            ys = [torch.empty(pt_ys.shape).cuda().half()]
            exe_module.run_with_tensors(inputs, ys)
            self.assertTrue(torch.allclose(pt_ys, ys[0], atol=1e-2, rtol=1e-2))
            print("Batch {} MHA verification pass".format(batch_size))

            if benchmark_ait:
                # Benchmark AIT
                time_per_iter_ms, time_std, _ = exe_module.benchmark_with_tensors(
                    inputs,
                    ys,
                    count=100,
                )
                _LOGGER.info("benchmark cross-attn time: {0}".format(time_per_iter_ms))

    def test_cross_attn(self):
        self._test_mha(batch_sizes=[1], seqlen=2, seqlen_kv=32, dim=512, num_heads=8)
        self._test_mha(
            batch_sizes=[128, 256, 512], seqlen=1, seqlen_kv=62, dim=512, num_heads=8
        )
        self._test_mha(
            batch_sizes=[1, 32, 64], seqlen=128, seqlen_kv=62, dim=512, num_heads=8
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
