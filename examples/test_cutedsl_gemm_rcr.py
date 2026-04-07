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
CuTeDSL gemm_rcr (no bias) via AIT compile_model test
======================================================

Validates the CuTeDSL backend for gemm_rcr by compiling an AIT graph
and checking numerical correctness against PyTorch.

Operation: Y[M, N] = X[M, K] @ W[N, K]^T
Equivalent: torch.nn.functional.linear(X, W, bias=None)

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:test_cutedsl_gemm_rcr
    buck run fbcode//aitemplate/AITemplate/examples:test_cutedsl_gemm_rcr -- --both
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
            f"gemm_rcr CuTeDSL requires SM80+ (A100/H100). Current GPU: SM{gpu_arch}"
        )

    print(f"  Detected GPU architecture: SM{gpu_arch}")
    return FBCUDA(arch=gpu_arch, **kwargs)


def build_gemm_rcr_graph(M, N, K, dtype="float16"):
    """Build AIT graph for gemm_rcr: Y[M,N] = X[M,K] @ W[N,K]^T."""
    X = Tensor(shape=[M, K], dtype=dtype, name="X", is_input=True)
    W = Tensor(shape=[N, K], dtype=dtype, name="W", is_input=True)

    Y = ops.gemm_rcr()(X, W)

    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    return Y


def run_test(M, N, K, use_cutedsl=False):
    """Compile and run gemm_rcr through AIT compile_model."""
    backend_name = "CuTeDSL" if use_cutedsl else "CUTLASS C++"
    print(f"\n  --- gemm_rcr ({backend_name}) M={M}, N={N}, K={K} ---")

    # PyTorch reference
    x_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_pt = torch.randn(N, K, device="cuda", dtype=torch.float16)
    y_pt = torch.nn.functional.linear(x_pt, w_pt, bias=None)

    # Build AIT graph
    target = _get_target(use_fp16_acc=False, use_cutedsl_gemm=use_cutedsl)
    logging.getLogger("aitemplate").setLevel(logging.DEBUG)

    with target:
        Y = build_gemm_rcr_graph(M, N, K)

    # Compile and run
    workdir_suffix = "cutedsl" if use_cutedsl else "cutlass"
    print(f"  Compiling with {backend_name} backend...")
    with compile_model(
        Y, target, "./tmp", f"gemm_rcr_{workdir_suffix}_{M}_{N}_{K}"
    ) as module:
        y_ait = torch.empty_like(y_pt)
        module.run_with_tensors(
            {"X": x_pt, "W": w_pt},
            {"Y": y_ait},
        )

        # Validate
        close = torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2)
        max_diff = (y_ait - y_pt).abs().max().item()
        assert close, f"Results mismatch! Max diff: {max_diff}"
        print(f"  Results match PyTorch: max diff = {max_diff:.6f}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="CuTeDSL gemm_rcr (no bias) via AIT compile_model test"
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
    print("CuTeDSL gemm_rcr (no bias) Test")
    print("=" * 60)
    print("Operation: Y[M,N] = X[M,K] @ W[N,K]^T")

    test_shapes = [
        (256, 512, 128),
        (128, 256, 64),
        (1, 1024, 512),
    ]

    for M, N, K in test_shapes:
        if args.both:
            run_test(M, N, K, use_cutedsl=False)
            run_test(M, N, K, use_cutedsl=True)
        else:
            run_test(M, N, K, use_cutedsl=args.use_cutedsl or True)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
