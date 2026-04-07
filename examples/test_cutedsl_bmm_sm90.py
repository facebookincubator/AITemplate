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
Direct test harness for BmmSm90Kernel (cutedsl_bmm_sm90.py).

Directly invokes the CuTeDSL SM90 BMM kernel via cute.compile + execute,
bypassing the full AITemplate compilation pipeline. Validates results
against PyTorch reference across all layout/variant combinations.

The kernel expects batch-last tensor ordering: A(M,K,B), B(N,K,B), C(M,N,B).
This test creates tensors in PyTorch batch-first format, computes the
reference, then permutes to batch-last for the CuTe kernel.

Requires SM90+ (H100 / Hopper).

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:test_cutedsl_bmm_sm90
"""

import sys

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from aitemplate.backend.cuda.gemm_universal.cutedsl_bmm_sm90 import BmmSm90Kernel
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# Helpers
# =============================================================================


def make_cute_tensor(t):
    """Convert a PyTorch CUDA tensor to a CuTe tensor with dynamic modes.

    Marks the innermost (stride-1) mode as dynamic. For compact tensors,
    this single call makes all dependent strides dynamic as well.

    Skipped when any dimension has size 1 (e.g. B=1 batch), because
    mark_compact_shape_dynamic cannot verify compact stride ordering
    for size-1 dimensions in permuted views.
    """
    ct = from_dlpack(t, assumed_align=16)
    if all(s > 1 for s in t.shape):
        innermost_mode = t.dim_order()[0]
        ct = ct.mark_compact_shape_dynamic(
            mode=innermost_mode,
            stride_order=t.dim_order(),
            divisibility=1,
        )
    return ct


def to_batch_last_a(t, a_row_major):
    """Permute A from batch-first to batch-last (M, K, B).

    Row-major A: (B, M, K) -> (M, K, B) via permute(1, 2, 0)
    Col-major A: (B, K, M) -> (M, K, B) via permute(2, 1, 0)
    """
    return t.permute(1, 2, 0) if a_row_major else t.permute(2, 1, 0)


def to_batch_last_b(t, b_row_major):
    """Permute B from batch-first to batch-last (N, K, B).

    Row-major B: (B, K, N) -> (N, K, B) via permute(2, 1, 0)
    Col-major B: (B, N, K) -> (N, K, B) via permute(1, 2, 0)
    """
    return t.permute(2, 1, 0) if b_row_major else t.permute(1, 2, 0)


def to_batch_last_c(t):
    """Permute C/D from batch-first (B, M, N) to batch-last (M, N, B)."""
    return t.permute(1, 2, 0)


def get_cu_stream():
    """Get CUDA driver stream from current PyTorch stream."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# =============================================================================
# Layout configs: (name, a_row_major, b_row_major, A_shape_fn, B_shape_fn, ref_fn)
# =============================================================================


def _make_configs():
    """Build layout test configs.

    Each config: (name, a_row_major, b_row_major,
                  A_shape(B,M,N,K), B_shape(B,M,N,K), ref_fn(a,b))
    """
    return [
        (
            "rrr",
            True,
            True,
            lambda B, M, N, K: (B, M, K),
            lambda B, M, N, K: (B, K, N),
            lambda a, b: torch.bmm(a, b),
        ),
        (
            "ccr",
            False,
            False,
            lambda B, M, N, K: (B, K, M),
            lambda B, M, N, K: (B, N, K),
            lambda a, b: torch.bmm(a.transpose(-2, -1), b.transpose(-2, -1)),
        ),
        (
            "rcr",
            True,
            False,
            lambda B, M, N, K: (B, M, K),
            lambda B, M, N, K: (B, N, K),
            lambda a, b: torch.bmm(a, b.transpose(-2, -1)),
        ),
    ]


# Shape configs: (name, B, M, N, K)
_SHAPES = [
    ("aligned", 2, 256, 512, 128),
    ("medium", 4, 512, 256, 256),
    ("large_batch", 16, 128, 128, 64),
    ("small", 1, 128, 128, 64),
]


# =============================================================================
# Core test runner
# =============================================================================


def run_test(
    name,
    a_row_major,
    b_row_major,
    has_d,
    B,
    M,
    N,
    K,
    a_shape,
    b_shape,
    ref_fn,
    atol=1e-2,
    rtol=1e-2,
):
    """Run a single BmmSm90Kernel test case."""
    add_str = "_add" if has_d else ""
    test_id = f"bmm_{name}{add_str} B={B} M={M} N={N} K={K}"

    # Create kernel
    kernel = BmmSm90Kernel(
        tile_m=128,
        tile_n=128,
        a_row_major=a_row_major,
        b_row_major=b_row_major,
        has_d=has_d,
    )

    # Create PyTorch tensors (batch-first, standard PyTorch convention)
    a_pt = torch.randn(*a_shape, device="cuda", dtype=torch.float16)
    b_pt = torch.randn(*b_shape, device="cuda", dtype=torch.float16)
    c_pt = torch.zeros(B, M, N, device="cuda", dtype=torch.float16)
    d_pt = (
        torch.randn(B, M, N, device="cuda", dtype=torch.float16)
        if has_d
        else torch.zeros(B, M, N, device="cuda", dtype=torch.float16)
    )

    # PyTorch reference (batch-first)
    y_ref = ref_fn(a_pt, b_pt)
    if has_d:
        y_ref = y_ref + d_pt

    # Permute to batch-last for the kernel: A(M,K,B), B(N,K,B), C/D(M,N,B).
    # These are views sharing memory with the batch-first tensors.
    a_bl = to_batch_last_a(a_pt, a_row_major)
    b_bl = to_batch_last_b(b_pt, b_row_major)
    c_bl = to_batch_last_c(c_pt)
    d_bl = to_batch_last_c(d_pt)

    # Convert to CuTe tensors
    a_cute = make_cute_tensor(a_bl)
    b_cute = make_cute_tensor(b_bl)
    c_cute = make_cute_tensor(c_bl)
    d_cute = make_cute_tensor(d_bl)

    cu_stream = get_cu_stream()

    # JIT compile
    compiled = cute.compile(
        kernel,
        a_cute,
        b_cute,
        c_cute,
        d_cute,
        B,
        M,
        N,
        K,
        cu_stream,
    )

    # Execute
    compiled(
        a_cute,
        b_cute,
        c_cute,
        d_cute,
        B,
        M,
        N,
        K,
        cu_stream,
    )
    torch.cuda.synchronize()

    # Validate — c_pt (batch-first) shares memory with c_bl (batch-last),
    # so it already has the kernel output in batch-first layout.
    max_diff = (c_pt - y_ref).abs().max().item()
    passed = torch.allclose(c_pt, y_ref, atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_id}  (max_diff={max_diff:.6f})")
    return passed


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("BmmSm90Kernel Direct Test Harness")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required")
        sys.exit(1)

    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_arch = cc_major * 10 + cc_minor
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} (SM{gpu_arch})")

    if gpu_arch < 90:
        print(f"ERROR: SM90+ required for Hopper TMA/WGMMA, got SM{gpu_arch}")
        sys.exit(1)

    configs = _make_configs()
    total = 0
    passed = 0
    failed_tests = []

    # Test plain BMM (has_d=False) for all layouts and shapes
    for (
        layout_name,
        a_row_major,
        b_row_major,
        a_shape_fn,
        b_shape_fn,
        ref_fn,
    ) in configs:
        print(f"\n--- bmm_{layout_name} (plain) ---")
        for shape_name, B, M, N, K in _SHAPES:
            a_shape = a_shape_fn(B, M, N, K)
            b_shape = b_shape_fn(B, M, N, K)
            total += 1
            ok = run_test(
                layout_name,
                a_row_major,
                b_row_major,
                has_d=False,
                B=B,
                M=M,
                N=N,
                K=K,
                a_shape=a_shape,
                b_shape=b_shape,
                ref_fn=ref_fn,
            )
            if ok:
                passed += 1
            else:
                failed_tests.append(
                    f"bmm_{layout_name} {shape_name} B={B} M={M} N={N} K={K}"
                )

    # Test BMM + residual add (has_d=True) for all layouts and shapes
    for (
        layout_name,
        a_row_major,
        b_row_major,
        a_shape_fn,
        b_shape_fn,
        ref_fn,
    ) in configs:
        print(f"\n--- bmm_{layout_name}_add (residual) ---")
        for shape_name, B, M, N, K in _SHAPES:
            a_shape = a_shape_fn(B, M, N, K)
            b_shape = b_shape_fn(B, M, N, K)
            total += 1
            ok = run_test(
                layout_name,
                a_row_major,
                b_row_major,
                has_d=True,
                B=B,
                M=M,
                N=N,
                K=K,
                a_shape=a_shape,
                b_shape=b_shape,
                ref_fn=ref_fn,
            )
            if ok:
                passed += 1
            else:
                failed_tests.append(
                    f"bmm_{layout_name}_add {shape_name} B={B} M={M} N={N} K={K}"
                )

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} passed")
    if failed_tests:
        print(f"\nFailed tests:")
        for t in failed_tests:
            print(f"  - {t}")
    else:
        print("All tests passed!")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
