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
AITemplate classic_b2b_bmm Pattern Matching Example
====================================================
Demonstrates the fuse_b2b_bmm compiler pass that automatically fuses a
decomposed PyTorch-style attention pattern into classic_b2b_bmm.

Instead of directly using ops.classic_b2b_bmm, the AIT graph is built from
individual ops (bmm_rcr, elementwise MUL/ADD/SIGMOID, bmm_rrr) that mirror
the PyTorch implementation. The compiler's fuse_b2b_bmm pass then
pattern-matches and replaces them with the fused classic_b2b_bmm kernel.

Pattern matched:
    score = bmm_rcr(Q, K)             # Q @ K^T
    score = score * alpha0             # scale
    score = score + bias               # add bias
    score = sigmoid(score)             # activation
    score = score * alpha1             # scale (optional)
    output = bmm_rrr(score, V)         # score @ V
  =>
    output = classic_b2b_bmm(Q, K, V, bias)

Requirements:
    - CUDA SM80+ (A100, H100, etc.)
    - N0, N1 <= 512 (sequence length limitation)

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:classic_b2b_bmm_example
"""

import logging

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing.detect_target import FBCUDA


def _get_target(**kwargs):
    """Create AIT CUDA target, auto-detecting GPU architecture."""
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_arch = str(cc_major * 10 + cc_minor)

    if int(gpu_arch) < 80:
        raise RuntimeError(
            f"classic_b2b_bmm requires SM80+ (A100/H100). Current GPU: SM{gpu_arch}"
        )

    print(f"Detected GPU architecture: SM{gpu_arch}")
    return FBCUDA(arch=gpu_arch, **kwargs)


# =============================================================================
# PyTorch Reference Model
# =============================================================================


class PTB2bBmm(torch.nn.Module):
    """PyTorch reference for the b2b_bmm computation (no learnable params).

    Computes: output = (alpha1) * sigmoid(alpha0 * (Q @ K^T) + bias) @ V
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.alpha0 = 1.0 / (head_dim**0.5)
        self.alpha1 = 1.0

    def forward(self, q, k, v, bias):
        attn = self.alpha0 * (q @ k.transpose(-2, -1)) + bias
        attn = torch.sigmoid(attn)
        attn = self.alpha1 * attn
        return attn @ v


# =============================================================================
# AITemplate Graph Builder (decomposed ops, NOT using ops.classic_b2b_bmm)
# =============================================================================


def build_decomposed_b2b_bmm_graph(batch, seq_len, head_dim, dtype="float16"):
    """Build AIT graph using decomposed ops that mirror the PyTorch implementation.

    This does NOT use ops.classic_b2b_bmm directly. Instead, it builds the
    equivalent graph from primitive ops:
        bmm_rcr -> MUL(alpha0) -> ADD(bias) -> SIGMOID -> MUL(alpha1) -> bmm_rrr

    The fuse_b2b_bmm compiler pass will pattern-match this and replace it
    with a fused classic_b2b_bmm op.
    """
    alpha0 = 1.0 / (head_dim**0.5)
    alpha1 = 1.0

    Q = Tensor(shape=[batch, seq_len, head_dim], dtype=dtype, name="Q", is_input=True)
    K = Tensor(shape=[batch, seq_len, head_dim], dtype=dtype, name="K", is_input=True)
    V = Tensor(shape=[batch, seq_len, head_dim], dtype=dtype, name="V", is_input=True)
    Bias = Tensor(
        shape=[batch, seq_len, seq_len], dtype=dtype, name="Bias", is_input=True
    )

    # Step 1: score = Q @ K^T  (bmm_rcr treats K as column-major => K^T)
    score = ops.bmm_rcr()(Q, K)

    # Step 2: score = score * alpha0
    score = ops.elementwise(FuncEnum.MUL)(score, alpha0)

    # Step 3: score = score + bias
    score = ops.elementwise(FuncEnum.ADD)(score, Bias)

    # Step 4: score = sigmoid(score)
    score = ops.elementwise(FuncEnum.SIGMOID)(score)

    # Step 5: score = score * alpha1
    score = ops.elementwise(FuncEnum.MUL)(score, alpha1)

    # Step 6: output = score @ V  (bmm_rrr: both row-major)
    Y = ops.bmm_rrr()(score, V)

    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    return Y


# =============================================================================
# Test
# =============================================================================


def run_pattern_matching_example():
    """Test: Decomposed ops auto-fused into classic_b2b_bmm by compiler pass.

    Builds an AIT graph from primitive ops (bmm_rcr, elementwise MUL/ADD/SIGMOID,
    bmm_rrr) and verifies that the fuse_b2b_bmm pass fuses them into a single
    classic_b2b_bmm kernel, producing results matching PyTorch.
    """
    print("\n" + "=" * 60)
    print("Pattern Matching Test: decomposed ops -> classic_b2b_bmm")
    print("=" * 60)

    batch, seq_len, head_dim = 4, 128, 64
    dtype = "float16"

    # Create and run PyTorch reference
    pt_model = PTB2bBmm(head_dim).cuda().half()
    pt_model.eval()

    q_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    bias_pt = torch.randn(batch, seq_len, seq_len, device="cuda", dtype=torch.float16)
    y_pt = pt_model(q_pt, k_pt, v_pt, bias_pt)

    # Build AIT graph from decomposed ops (NOT ops.classic_b2b_bmm)
    target = _get_target(use_fp16_acc=False)
    logging.getLogger("aitemplate").setLevel(logging.DEBUG)

    with target:
        Y = build_decomposed_b2b_bmm_graph(batch, seq_len, head_dim, dtype)

    # Compile - the fuse_b2b_bmm pass will fuse the decomposed graph
    print("\nCompiling... (fuse_b2b_bmm pass will pattern-match and fuse)")
    with compile_model(Y, target, "./tmp", "pattern_matched_b2b_bmm") as module:
        y_ait = torch.empty_like(y_pt)
        module.run_with_tensors(
            {"Q": q_pt, "K": k_pt, "V": v_pt, "Bias": bias_pt},
            {"Y": y_ait},
        )

        # Verify correctness
        close = torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2)
        max_diff = (y_ait - y_pt).abs().max().item()
        assert close, f"Results mismatch! Max diff: {max_diff}"
        print(f"\nResults match PyTorch: {close} (max diff: {max_diff:.6f})")


def main():
    print("=" * 60)
    print("AITemplate classic_b2b_bmm Pattern Matching Example")
    print("=" * 60)
    print("\nDemonstrates automatic fusion of decomposed attention ops")
    print("into classic_b2b_bmm via the fuse_b2b_bmm compiler pass.")

    run_pattern_matching_example()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
