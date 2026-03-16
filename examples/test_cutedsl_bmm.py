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
"""Tests for CuTeDSL backends of BMM operations.

ALL TESTS PASSING ON H100 (SM90)! ✓

This test file covers multiple BMM layout combinations with CuTeDSL kernels:
  - bmm_rcr:     C[B,M,N] = A[B,M,K] @ B[B,N,K]^T (IDEAL - no transpose needed) ✓
  - bmm_rcr_add: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T + D[B,M,N] ✓
  - bmm_rrr:     C[B,M,N] = A[B,M,K] @ B[B,K,N] ✓
  - bmm_ccr:     C[B,M,N] = A[B,K,M]^T @ B[B,N,K]^T ✓
  - bmm_rrr_add: C[B,M,N] = A[B,M,K] @ B[B,K,N] + D[B,M,N] ✓
  - bmm_ccr_add: C[B,M,N] = A[B,K,M]^T @ B[B,N,K]^T + D[B,M,N] ✓

Key fix: The tensor descriptor format was corrected to:
  - dynamic_shapes[0] = B_dim (batch size)
  - dynamic_shapes[1] = first inner dimension extent (M, K, or N)
  - dynamic_strides[0] = batch stride

MMA Layout Requirements:
  - A operand: [M, K] with K contiguous
  - B operand: [N, K] with K contiguous

Layout analysis:
  - RCR: A[M,K] K-contiguous ✓, B[N,K] K-contiguous ✓ (IDEAL, no transpose)
  - RRR: A[M,K] K-contiguous ✓, B[K,N] needs logical transpose
  - CCR: A[K,M] needs logical transpose, B[N,K] K-contiguous ✓

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:test_cutedsl_bmm
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


# All tests passing on H100 (SM90)!
class CuTeDSLBmmTest(unittest.TestCase):
    """Tests for basic BMM operations (bmm_ccr, bmm_rrr).

    These tests use layouts that require logical transpose handling.
    Fixed by correcting the tensor descriptor format in the wrapper.
    """

    def test_bmm_rrr(self):
        """Test bmm_rrr: C[B,M,N] = A[B,M,K] @ B[B,K,N]"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        A = Tensor(shape=[B, M, K], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, K, N], dtype=dtype, name="W", is_input=True)
        Y = ops.bmm_rrr()(A, W)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_rrr") as module:
            a_pt = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

            # PyTorch reference: standard batched matmul
            y_ref = torch.bmm(a_pt, w_pt)

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_rrr: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )

    def test_bmm_ccr(self):
        """Test bmm_ccr: C[B,M,N] = A[B,K,M]^T @ B[B,N,K]^T"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        # A is col-major [B, K, M], B is col-major [B, N, K]
        A = Tensor(shape=[B, K, M], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="W", is_input=True)
        Y = ops.bmm_ccr()(A, W)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_ccr") as module:
            a_pt = torch.randn(B, K, M, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, N, K, device="cuda", dtype=torch.float16)

            # PyTorch reference:
            # bmm_ccr: A^T @ W^T = transpose(A, -2, -1) @ transpose(W, -2, -1)
            a_t = a_pt.transpose(-2, -1)  # [B, M, K]
            w_t = w_pt.transpose(-2, -1)  # [B, K, N]
            y_ref = torch.bmm(a_t, w_t)

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_ccr: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )


# All tests passing on H100 (SM90)!
class CuTeDSLBmmAddTest(unittest.TestCase):
    """Tests for BMM with residual add (bmm_ccr_add, bmm_rrr_add).

    Fixed by correcting the tensor descriptor format in the wrapper.
    """

    def test_bmm_rrr_add(self):
        """Test bmm_rrr_add: C[B,M,N] = A[B,M,K] @ B[B,K,N] + D[B,M,N]"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        A = Tensor(shape=[B, M, K], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, K, N], dtype=dtype, name="W", is_input=True)
        D = Tensor(shape=[B, M, N], dtype=dtype, name="D", is_input=True)
        Y = ops.bmm_rrr_add()(A, W, D)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_rrr_add") as module:
            a_pt = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
            d_pt = torch.randn(B, M, N, device="cuda", dtype=torch.float16)

            # PyTorch reference: bmm + add
            y_ref = torch.bmm(a_pt, w_pt) + d_pt

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt, "D": d_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_rrr_add: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )

    def test_bmm_ccr_add(self):
        """Test bmm_ccr_add: C[B,M,N] = A[B,K,M]^T @ B[B,N,K]^T + D[B,M,N]"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        # A is col-major [B, K, M], B is col-major [B, N, K]
        A = Tensor(shape=[B, K, M], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="W", is_input=True)
        D = Tensor(shape=[B, M, N], dtype=dtype, name="D", is_input=True)
        Y = ops.bmm_ccr_add()(A, W, D)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_ccr_add") as module:
            a_pt = torch.randn(B, K, M, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, N, K, device="cuda", dtype=torch.float16)
            d_pt = torch.randn(B, M, N, device="cuda", dtype=torch.float16)

            # PyTorch reference:
            # bmm_ccr_add: A^T @ W^T + D
            a_t = a_pt.transpose(-2, -1)  # [B, M, K]
            w_t = w_pt.transpose(-2, -1)  # [B, K, N]
            y_ref = torch.bmm(a_t, w_t) + d_pt

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt, "D": d_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_ccr_add: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )


# =============================================================================
# Tests for bmm_rcr layout - WORKING (no transpose needed)
# =============================================================================


class CuTeDSLBmmRcrTest(unittest.TestCase):
    """Tests for BMM RCR layout - IDEAL for CuTeDSL MMA.

    bmm_rcr: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T

    Why this works:
    - A[M,K] row-major has K contiguous ✓ (matches MMA A operand)
    - B[N,K] col-major has K contiguous ✓ (matches MMA B operand)
    - Both operands match MMA requirements - NO TRANSPOSE NEEDED!

    This is equivalent to torch.bmm(A, B.transpose(-2, -1)).
    """

    def test_bmm_rcr(self):
        """Test bmm_rcr: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        # A is row-major [B, M, K], B is col-major [B, N, K]
        A = Tensor(shape=[B, M, K], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="W", is_input=True)
        Y = ops.bmm_rcr()(A, W)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_rcr") as module:
            a_pt = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, N, K, device="cuda", dtype=torch.float16)

            # PyTorch reference: A @ W^T
            y_ref = torch.bmm(a_pt, w_pt.transpose(-2, -1))

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_rcr: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )


class CuTeDSLBmmRcrAddTest(unittest.TestCase):
    """Tests for BMM RCR with residual add - WORKING.

    bmm_rcr_add: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T + D[B,M,N]
    """

    def test_bmm_rcr_add(self):
        """Test bmm_rcr_add: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T + D[B,M,N]"""
        B, M, N, K = 2, 256, 512, 128
        dtype = "float16"

        A = Tensor(shape=[B, M, K], dtype=dtype, name="A", is_input=True)
        W = Tensor(shape=[B, N, K], dtype=dtype, name="W", is_input=True)
        D = Tensor(shape=[B, M, N], dtype=dtype, name="D", is_input=True)
        Y = ops.bmm_rcr_add()(A, W, D)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True

        target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=True)
        with compile_model(Y, target, "./tmp", "test_cutedsl_bmm_rcr_add") as module:
            a_pt = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
            w_pt = torch.randn(B, N, K, device="cuda", dtype=torch.float16)
            d_pt = torch.randn(B, M, N, device="cuda", dtype=torch.float16)

            # PyTorch reference: A @ W^T + D
            y_ref = torch.bmm(a_pt, w_pt.transpose(-2, -1)) + d_pt

            y_ait = torch.empty(B, M, N, device="cuda", dtype=torch.float16)
            module.run_with_tensors(
                {"A": a_pt, "W": w_pt, "D": d_pt},
                {"Y": y_ait},
            )

            self.assertTrue(
                torch.allclose(y_ait, y_ref, atol=1e-1, rtol=1e-1),
                f"bmm_rcr_add: max diff = {(y_ait - y_ref).abs().max().item():.6f}",
            )


if __name__ == "__main__":
    unittest.main()
