#  Copyright (c) Meta Platform, Inc. and its affiliates.
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
import math
import unittest

import numpy as np

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import logger, shape_utils


class NewGELUActivation(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class T5DenseGatedGeluDense(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.wo = torch.nn.Linear(d_ff, d_model, bias=False)
        self.gelu_act = NewGELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


@unittest.skipIf(detect_target()._arch == "75", "DualGemm not supported on sm75.")
class DUALGEMMTestCase(unittest.TestCase):
    def _test_dual_gemm(self, M=4096, N=4096, K=8192, fast_gelu=False, benchmark=False):
        target = detect_target(use_fp16_acc=False)
        X = Tensor(shape=[M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype="float16", name="input_1", is_input=True)
        B = Tensor(shape=[N, K], dtype="float16", name="input_2", is_input=True)
        if fast_gelu:
            OP = ops.dual_gemm_rcr_fast_gelu()
        else:
            OP = ops.dual_gemm_rcr_silu()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "dual_gemm")
        X_pt = torch.randn(M, K).cuda().half() * 0.01
        W_pt = torch.randn(N, K).cuda().half()
        B_pt = torch.randn(N, K).cuda().half()

        def pt_func(X_pt, W_pt, B_pt):
            Y_pt1 = torch.nn.functional.linear(X_pt, W_pt)
            Y_pt2 = torch.nn.functional.linear(X_pt, B_pt)
            if fast_gelu:
                gelu_act = NewGELUActivation()
            else:
                gelu_act = torch.nn.functional.silu
            Y_pt = gelu_act(Y_pt1) * Y_pt2
            return Y_pt

        Y_pt = pt_func(X_pt, W_pt, B_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = torch.empty([M, N]).cuda().half()
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        if benchmark:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark AIT
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            logger.info(__file__, f"AIT GEMMxGEMM time: {time_per_iter_ms:.5f}ms")
            # Benchmark PT
            from aitemplate.testing.benchmark_pt import benchmark_torch_function

            func = pt_func
            args = (X_pt, W_pt, B_pt)
            duration = benchmark_torch_function(100, func, *args)
            logger.info(__file__, f"PT GEMMxGEMM Time: {duration:.5f}ms")

    def test_dual_gemm(self):
        for fast_gelu in [True, False]:
            self._test_dual_gemm(M=128, N=128, K=256, fast_gelu=fast_gelu)
            self._test_dual_gemm(M=1024, N=1024, K=2048, fast_gelu=fast_gelu)
            self._test_dual_gemm(M=4096, N=4096, K=8192, fast_gelu=fast_gelu)

    def _test_t5block(
        self,
        Ms,
        d_model=1024,
        d_ff=2048,
        use_fp16_acc=False,
    ):

        pt_mod = T5DenseGatedGeluDense(d_model=d_model, d_ff=d_ff).cuda().half()
        pt_mod = pt_mod.eval()

        pt_params = dict(pt_mod.named_parameters())
        params_ait = {}
        for key, arr in pt_params.items():
            print(key, arr.shape)
            params_ait[key.replace(".", "_").replace("out_proj", "proj")] = arr

        ait_mod = nn.T5DenseGatedGeluDense(
            in_channels=d_model,
            out_channels=d_ff,
        )
        ait_mod.name_parameter_tensor()

        M_dim = shape_utils.gen_int_var_min_max(Ms, name="Mdim")
        inputs_ait = Tensor([M_dim, d_model], name="input0", is_input=True)
        Y = ait_mod(inputs_ait)
        mark_output(Y)
        target = detect_target(use_fp16_acc=False)
        exe_module = compile_model(Y, target, "./tmp", "t5block")
        for name, weight in params_ait.items():
            exe_module.set_constant_with_tensor(name, weight)

        for m in Ms:
            input_pt = torch.randn([m, d_model]).cuda().half()
            pt_ys = pt_mod(input_pt)
            print("pt output:", pt_ys.shape)

            inputs = [input_pt]
            ys = [torch.empty(pt_ys.shape).cuda().half()]
            exe_module.run_with_tensors(inputs, ys)
            eps = 1e-2
            np.testing.assert_allclose(
                pt_ys.detach().cpu().numpy(),
                ys[0].cpu().numpy(),
                atol=eps,
                rtol=eps,
            )
            print("M = {} t5 verification pass".format(m))

    def test_t5block(self):
        self._test_t5block(Ms=[1024, 2048, 4096])


if __name__ == "__main__":
    unittest.main()
