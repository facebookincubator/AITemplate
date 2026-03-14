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
CuTeDSL gemm_rcr_bias via AIT compile_model Example
=====================================================

Demonstrates end-to-end compilation of a PyTorch linear layer (GEMM + bias)
through AITemplate's compile_model pipeline with the CuTeDSL backend.

The flow:
1. Define a simple PyTorch model using torch.nn.functional.linear
2. Build the equivalent AIT graph using ops.gemm_rcr_bias
3. Compile with CuTeDSL backend (use_cutedsl_gemm=True) which:
   - AOT-compiles a CuTeDSL kernel via cute.compile() + export_to_c()
   - Produces .h header + .o object with embedded cubin
   - Wraps it in a thin C++ shim with the standard AIT function signature
   - Links everything into a .so module
4. Run the compiled module and validate against PyTorch reference

Operation: Y[M, N] = X[M, K] @ W[N, K]^T + Bias[N]
Equivalent: torch.nn.functional.linear(X, W, bias=Bias)

Requirements:
    - CUDA SM80+ (A100, H100, etc.)

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:cutedsl_gemm_rcr_bias_compile_model_example
    buck run fbcode//aitemplate/AITemplate/examples:cutedsl_gemm_rcr_bias_compile_model_example -- --both
"""

import argparse
import logging

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing.detect_target import FBCUDA


def _get_target(**kwargs):
    """Create AIT CUDA target, auto-detecting GPU architecture."""
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_arch = str(cc_major * 10 + cc_minor)

    if int(gpu_arch) < 80:
        raise RuntimeError(
            f"gemm_rcr_bias CuTeDSL requires SM80+ (A100/H100). Current GPU: SM{gpu_arch}"
        )

    print(f"  Detected GPU architecture: SM{gpu_arch}")
    return FBCUDA(arch=gpu_arch, **kwargs)


# =============================================================================
# PyTorch Reference Model
# =============================================================================


class PTLinear(torch.nn.Module):
    """PyTorch reference: Y = X @ W^T + Bias (torch.nn.functional.linear)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# AITemplate Graph Builder
# =============================================================================


def build_gemm_rcr_bias_graph(M, N, K, dtype="float16"):
    """Build AIT graph for gemm_rcr_bias: Y[M,N] = X[M,K] @ W[N,K]^T + Bias[N].

    This uses ops.gemm_rcr_bias directly, which is the AIT op corresponding
    to torch.nn.functional.linear(X, W, bias).
    """
    X = Tensor(shape=[M, K], dtype=dtype, name="X", is_input=True)
    W = Tensor(shape=[N, K], dtype=dtype, name="W", is_input=True)
    Bias = Tensor(shape=[N], dtype=dtype, name="Bias", is_input=True)

    Y = ops.gemm_rcr_bias()(X, W, Bias)

    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    return Y


# =============================================================================
# Test
# =============================================================================


def run_example(use_cutedsl=False):
    """Compile and run gemm_rcr_bias through AIT compile_model.

    Parameters
    ----------
    use_cutedsl : bool
        If True, use CuTeDSL backend instead of CUTLASS C++ templates.
    """
    backend_name = "CuTeDSL" if use_cutedsl else "CUTLASS C++"
    print("\n" + "=" * 60)
    print(f"gemm_rcr_bias via compile_model ({backend_name})")
    print("=" * 60)

    M, N, K = 256, 512, 128
    dtype = "float16"

    print(f"  M={M}, N={N}, K={K}")
    print(f"  Operation: Y[{M},{N}] = X[{M},{K}] @ W[{N},{K}]^T + Bias[{N}]")

    # --- PyTorch reference ---
    pt_model = PTLinear(K, N).cuda().half()
    pt_model.eval()

    x_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        y_pt = pt_model(x_pt)

    # Extract weight and bias from the PyTorch module
    w_pt = pt_model.linear.weight.detach()  # shape [N, K]
    bias_pt = pt_model.linear.bias.detach()  # shape [N]

    # --- Build AIT graph ---
    target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=use_cutedsl)
    logging.getLogger("aitemplate").setLevel(logging.DEBUG)

    with target:
        Y = build_gemm_rcr_bias_graph(M, N, K, dtype)

    # --- Compile ---
    workdir_suffix = "cutedsl" if use_cutedsl else "cutlass"
    print(f"\n  Compiling with {backend_name} backend...")
    with compile_model(Y, target, "./tmp", f"gemm_rcr_bias_{workdir_suffix}") as module:
        y_ait = torch.empty_like(y_pt)
        module.run_with_tensors(
            {"X": x_pt, "W": w_pt, "Bias": bias_pt},
            {"Y": y_ait},
        )

        # --- Validate ---
        close = torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2)
        max_diff = (y_ait - y_pt).abs().max().item()
        assert close, f"Results mismatch! Max diff: {max_diff}"
        print(f"\n  Results match PyTorch: {close} (max diff: {max_diff:.6f})")
        print("  Numerical verification passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CuTeDSL gemm_rcr_bias via AIT compile_model example"
    )
    parser.add_argument(
        "--use-cutedsl",
        action="store_true",
        default=False,
        help="Use CuTeDSL backend instead of CUTLASS C++ templates",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        default=False,
        help="Run with both CUTLASS C++ and CuTeDSL backends",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CuTeDSL gemm_rcr_bias via AIT compile_model Example")
    print("=" * 60)
    print("\nDemonstrates: PyTorch linear -> AIT gemm_rcr_bias -> CuTeDSL AOT")
    print("Operation: Y[M,N] = X[M,K] @ W[N,K]^T + Bias[N]")

    if args.both:
        run_example(use_cutedsl=False)
        run_example(use_cutedsl=True)
    else:
        run_example(use_cutedsl=args.use_cutedsl)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
