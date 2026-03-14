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
Tests for CuTeDSL backends of gemm_rcr_bias activation/broadcast variants.

Covers:
  - gemm_rcr_bias_relu:        ReLU(A @ B^T + Bias)
  - gemm_rcr_bias_sigmoid:     Sigmoid(A @ B^T + Bias)
  - gemm_rcr_bias_swish:       SiLU(A @ B^T + Bias)
  - gemm_rcr_bias_sigmoid_mul: Sigmoid(A @ B^T + Bias) * D0
  - gemm_rcr_bias_mul_add:     (A @ B^T + Bias) * D0 + D1

Each test builds an AIT graph using the corresponding op, compiles with the
CuTeDSL backend (use_cutedsl_gemm=True), runs the compiled module, and
validates against the PyTorch reference.

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:test_cutedsl_gemm_rcr_bias_activations
"""

import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing.detect_target import FBCUDA


def _get_target(**kwargs):
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_arch = str(cc_major * 10 + cc_minor)
    if int(gpu_arch) < 80:
        raise RuntimeError(f"SM80+ required, got SM{gpu_arch}")
    return FBCUDA(arch=gpu_arch, **kwargs)


class CuTeDSLGemmRcrBiasActivationTest(unittest.TestCase):
    """Tests for simple activation epilogues (3-input ops)."""

    def _test_activation(self, op_func, pt_activation, M=256, N=512, K=128):
        """Helper: build AIT graph, compile with CuTeDSL, compare to PyTorch."""
        dtype = "float16"

        X = Tensor(shape=[M, K], dtype=dtype, name="X", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="W", is_input=True)
        Bias = Tensor(shape=[N], dtype=dtype, name="Bias", is_input=True)
        Y = op_func()(X, W, Bias)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        op_name = op_func.__name__
        with compile_model(Y, target, "./tmp", f"test_cutedsl_{op_name}") as module:
            x_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(N, K, device="cuda", dtype=torch.float16)
            bias_pt = torch.randn(N, device="cuda", dtype=torch.float16)

            y_ref = pt_activation(torch.nn.functional.linear(x_pt, w_pt, bias=bias_pt))

            y_ait = torch.empty(M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"X": x_pt, "W": w_pt, "Bias": bias_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"{op_name}: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )

    def test_gemm_rcr_bias_relu(self):
        self._test_activation(ops.gemm_rcr_bias_relu, torch.relu)

    def test_gemm_rcr_bias_sigmoid(self):
        self._test_activation(ops.gemm_rcr_bias_sigmoid, torch.sigmoid)

    def test_gemm_rcr_bias_swish(self):
        self._test_activation(ops.gemm_rcr_bias_swish, torch.nn.functional.silu)


class CuTeDSLGemmRcrBiasBroadcastTest(unittest.TestCase):
    """Tests for broadcast epilogues (4-5 input ops)."""

    def test_gemm_rcr_bias_sigmoid_mul(self):
        """Sigmoid(A @ B^T + Bias) * D0"""
        M, N, K = 256, 512, 128
        dtype = "float16"

        X = Tensor(shape=[M, K], dtype=dtype, name="X", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="W", is_input=True)
        Bias = Tensor(shape=[N], dtype=dtype, name="Bias", is_input=True)
        D0 = Tensor(shape=[M, N], dtype=dtype, name="D0", is_input=True)
        Y = ops.gemm_rcr_bias_sigmoid_mul()(X, W, Bias, D0)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(
            Y, target, "./tmp", "test_cutedsl_gemm_rcr_bias_sigmoid_mul"
        ) as module:
            x_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(N, K, device="cuda", dtype=torch.float16)
            bias_pt = torch.randn(N, device="cuda", dtype=torch.float16)
            d0_pt = torch.randn(M, N, device="cuda", dtype=torch.float16)

            linear = torch.nn.functional.linear(x_pt, w_pt, bias=bias_pt)
            y_ref = torch.sigmoid(linear) * d0_pt

            y_ait = torch.empty(M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"X": x_pt, "W": w_pt, "Bias": bias_pt, "D0": d0_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"sigmoid_mul: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )

    def test_gemm_rcr_bias_mul_add(self):
        """(A @ B^T + Bias) * D0 + D1"""
        M, N, K = 256, 512, 128
        dtype = "float16"

        X = Tensor(shape=[M, K], dtype=dtype, name="X", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="W", is_input=True)
        Bias = Tensor(shape=[N], dtype=dtype, name="Bias", is_input=True)
        D0 = Tensor(shape=[M, N], dtype=dtype, name="D0", is_input=True)
        D1 = Tensor(shape=[M, N], dtype=dtype, name="D1", is_input=True)
        Y = ops.gemm_rcr_bias_mul_add()(X, W, Bias, D0, D1)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(
            Y, target, "./tmp", "test_cutedsl_gemm_rcr_bias_mul_add"
        ) as module:
            x_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(N, K, device="cuda", dtype=torch.float16)
            bias_pt = torch.randn(N, device="cuda", dtype=torch.float16)
            d0_pt = torch.randn(M, N, device="cuda", dtype=torch.float16)
            d1_pt = torch.randn(M, N, device="cuda", dtype=torch.float16)

            linear = torch.nn.functional.linear(x_pt, w_pt, bias=bias_pt)
            y_ref = linear * d0_pt + d1_pt

            y_ait = torch.empty(M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"X": x_pt, "W": w_pt, "Bias": bias_pt, "D0": d0_pt, "D1": d1_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"mul_add: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )


if __name__ == "__main__":
    unittest.main()
