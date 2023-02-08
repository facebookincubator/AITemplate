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

import numpy as np
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


class crossattentionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = 0

    def _test_mha(
        self,
        batch_sizes,
        seqlen=1,
        seqlen_kv=62,
        dim=4,
        num_heads=2,
        use_fp16_acc=False,
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

        ait_mod = nn.CrossAttention(
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
        exe_module = compile_model(
            Y, target, "./tmp", f"cross_attn_dynamic_{self.test_id}"
        )
        self.test_id += 1
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
            eps = 1e-2
            np.testing.assert_allclose(
                pt_ys.detach().cpu().numpy(),
                ys[0].cpu().numpy(),
                atol=eps,
                rtol=eps,
            )
            print("Batch {} MHA verification pass".format(batch_size))

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    def test_cross_attn(self):
        self._test_mha(batch_sizes=[1], seqlen=2, seqlen_kv=32, dim=512, num_heads=8)
        self._test_mha(
            batch_sizes=[128, 256, 512], seqlen=1, seqlen_kv=62, dim=512, num_heads=8
        )
        self._test_mha(
            batch_sizes=[1, 32, 64], seqlen=128, seqlen_kv=62, dim=512, num_heads=8
        )
        self._test_mha(batch_sizes=[128], seqlen=1, seqlen_kv=4, dim=16, num_heads=2)


if __name__ == "__main__":
    unittest.main()
