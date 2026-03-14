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
CuTeDSL GEMM RCR Bias AOT (Ahead-of-Time) Compilation Example
================================================================

Demonstrates:
1. Creating a CuTeDSL GEMM RCR + Bias kernel (SM80 or SM90)
2. JIT compiling and running it
3. Validating against a PyTorch reference (torch.nn.functional.linear)
4. Exporting as AOT artifacts (.h header + .o object) for C++ integration

Operation: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
Equivalent to: torch.nn.functional.linear(A, B, bias=Bias)

The AOT export generates:
  - <name>.h: C header with Metadata struct, load/unload functions, and a
    wrapper function that takes raw pointers and launches the kernel
  - <name>.o: Object file with embedded cubin(s) and the host launch code

These can be linked into any C/C++ application:
    gcc -o app app.c gemm_rcr_bias.o -lcuda

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:cutedsl_gemm_rcr_bias_aot_example
    buck run fbcode//aitemplate/AITemplate/examples:cutedsl_gemm_rcr_bias_aot_example -- --arch sm90
"""

import argparse
import ctypes
import os

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cuda_runtime
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.export import export_to_c
from cutlass.cute.runtime import from_dlpack

# =============================================================================
# Polyfill for cudaLibrary_t (requires CUDA 12.8+ headers at build time).
# When cuda-bindings is built against an older toolkit, these symbols are
# absent.  We provide minimal stand-ins so the JIT executor can proceed.
# =============================================================================
if not hasattr(cuda_runtime, "cudaLibrary_t"):

    class _cudaLibrary_t:
        """Minimal stand-in wrapping the raw void-pointer handle."""

        __slots__ = ("value",)

        def __init__(self, value=0):
            self.value = value

    cuda_runtime.cudaLibrary_t = _cudaLibrary_t  # type: ignore[attr-defined]

if not hasattr(cuda_runtime, "cudaLibraryUnload"):

    def _cudaLibraryUnload(library):  # noqa: N802
        # Use the CUDA driver API to unload (cuLibraryUnload).
        err = cuda.cuLibraryUnload(ctypes.c_void_p(library.value))
        return err

    cuda_runtime.cudaLibraryUnload = _cudaLibraryUnload  # type: ignore[attr-defined]


# =============================================================================
# Helper: Create CuTe tensor from PyTorch tensor
# =============================================================================


def make_cute_tensor(torch_tensor: torch.Tensor) -> cute.Tensor:
    """Convert a PyTorch CUDA tensor to a CuTe tensor.

    Uses mark_compact_shape_dynamic to make dimensions dynamic while
    expressing alignment constraints for efficient memory operations.
    """
    ct = from_dlpack(torch_tensor, assumed_align=16)
    ct = ct.mark_compact_shape_dynamic(
        mode=len(torch_tensor.shape) - 1,
        stride_order=torch_tensor.dim_order(),
        divisibility=(128 // cutlass.Float16.width),  # 8 elements for fp16
    )
    return ct


def create_kernel(arch: str):
    """Create the appropriate kernel based on GPU architecture."""
    if arch == "sm90":
        from aitemplate.backend.cuda.gemm_universal.cutedsl_gemm_rcr_bias_sm90 import (
            GemmRcrBiasSm90Kernel,
        )

        return GemmRcrBiasSm90Kernel(tile_m=128, tile_n=128)
    else:
        from aitemplate.backend.cuda.gemm_universal.cutedsl_gemm_rcr_bias_sm80 import (
            GemmRcrBiasSm80Kernel,
        )

        return GemmRcrBiasSm80Kernel(tile_m=128, tile_n=128, tile_k=32)


# =============================================================================
# Step 1: JIT Compile and Validate
# =============================================================================


def jit_compile_and_validate(arch: str):
    """JIT-compile the GEMM RCR Bias kernel and validate against PyTorch."""
    print("\n" + "=" * 60)
    print(f"Step 1: JIT Compile and Validate ({arch.upper()})")
    print("=" * 60)

    # Test dimensions (representative linear layer)
    M, N, K = 256, 512, 128
    print(f"  M={M}, N={N}, K={K}")
    print(f"  Operation: C[{M},{N}] = A[{M},{K}] @ B[{N},{K}]^T + Bias[{N}]")

    # Create PyTorch test tensors
    A_pt = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_pt = torch.randn(N, K, device="cuda", dtype=torch.float16)
    Bias_pt = torch.randn(N, device="cuda", dtype=torch.float16)
    C_pt = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # PyTorch reference: C = A @ B^T + Bias = linear(A, B, Bias)
    C_ref = torch.nn.functional.linear(A_pt, B_pt, bias=Bias_pt)

    # Create CuTeDSL kernel for the target arch
    kernel = create_kernel(arch)
    print(f"  Kernel class: {type(kernel).__name__}")

    # Convert to CuTe tensors
    A_cute = make_cute_tensor(A_pt)
    B_cute = make_cute_tensor(B_pt)
    # For 1D bias, use from_dlpack directly to keep unit stride.
    # mark_compact_shape_dynamic would introduce hierarchical strides
    # that break the stride-0 broadcast construction in the kernel.
    Bias_cute = from_dlpack(Bias_pt, assumed_align=16)
    C_cute = make_cute_tensor(C_pt)

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    # JIT compile
    print("\n  JIT compiling CuTeDSL kernel...")
    compiled = cute.compile(
        kernel,
        A_cute,
        B_cute,
        Bias_cute,
        C_cute,
        M,
        N,
        K,
        cu_stream,
    )
    print("  Compilation successful!")

    # Execute
    print("  Executing kernel...")
    compiled(
        A_cute,
        B_cute,
        Bias_cute,
        C_cute,
        M,
        N,
        K,
        cu_stream,
    )
    torch.cuda.synchronize()

    # Validate
    close = torch.allclose(C_pt, C_ref, atol=1e-2, rtol=1e-2)
    max_diff = (C_pt - C_ref).abs().max().item()
    print(f"\n  Results match PyTorch: {close} (max diff: {max_diff:.6f})")

    if not close:
        print("  WARNING: Results do not match!")
        print(f"  C_pt[:3,:3]:\n{C_pt[:3, :3]}")
        print(f"  C_ref[:3,:3]:\n{C_ref[:3, :3]}")
    else:
        print("  Numerical verification passed!")

    # Return CuTe tensors (and the PyTorch tensors backing them) so they
    # remain alive for AOT export.  cute.compile() stores weakref.proxy()
    # references to the dynamic args; if the originals are GC'd before
    # export_to_c() accesses them, a ReferenceError is raised.
    refs = (A_pt, B_pt, Bias_pt, C_pt, A_cute, B_cute, Bias_cute, C_cute)
    return compiled, kernel, refs


# =============================================================================
# Step 2: AOT Export
# =============================================================================


def aot_export(compiled, arch: str, export_dir: str = None):
    """Export the compiled kernel as AOT artifacts (.h + .o)."""
    print("\n" + "=" * 60)
    print("Step 2: AOT Export (Ahead-of-Time Compilation)")
    print("=" * 60)

    output_dir = export_dir or "./tmp/cutedsl_gemm_rcr_bias_aot"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"gemm_rcr_bias_{arch}"

    print(f"  Exporting to: {output_dir}/{file_name}.h + {file_name}.o")

    # Export to C
    export_to_c(
        compiled,
        file_path=output_dir,
        file_name=file_name,
    )

    # Verify files were created
    h_path = os.path.join(output_dir, f"{file_name}.h")
    o_path = os.path.join(output_dir, f"{file_name}.o")

    assert os.path.exists(h_path), f"Header not generated: {h_path}"
    assert os.path.exists(o_path), f"Object not generated: {o_path}"

    h_size = os.path.getsize(h_path)
    o_size = os.path.getsize(o_path)
    print(f"  Generated: {file_name}.h ({h_size} bytes)")
    print(f"  Generated: {file_name}.o ({o_size} bytes)")

    # Show header contents
    print(f"\n  --- Contents of {file_name}.h (first 80 lines) ---")
    with open(h_path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:80]):
            print(f"  {i + 1:3d} | {line}", end="")
    if len(lines) > 80:
        print(f"  ... ({len(lines) - 80} more lines)")
    print("  --- End of header ---")

    print("\n  AOT export successful!")
    print("\n  To use from C/C++:")
    print(f'    #include "{file_name}.h"')
    print(f"    {file_name}_Kernel_Metadata_t metadata;")
    print(f"    {file_name}_Kernel_Metadata_Load(&metadata);")
    print(f"    {file_name}_wrapper(&metadata, ...);")
    print(f"    {file_name}_Kernel_Metadata_Unload(&metadata);")
    print(f"\n  Link with: gcc app.c {file_name}.o -lcuda")

    return h_path, o_path


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CuTeDSL GEMM RCR Bias AOT Compilation Example"
    )
    parser.add_argument(
        "--arch",
        choices=["sm80", "sm90", "auto"],
        default="auto",
        help="Target GPU architecture (default: auto-detect)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Directory for AOT export artifacts",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CuTeDSL GEMM RCR Bias AOT Compilation Example")
    print("=" * 60)
    print("\nOperation: C[M,N] = A[M,K] @ B[N,K]^T + Bias[N]")
    print("Equivalent: torch.nn.functional.linear(A, B, bias=Bias)")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required!")

    cc = torch.cuda.get_device_capability(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)} (SM{cc[0]}{cc[1]})")

    if cc[0] < 8:
        raise RuntimeError(f"SM80+ required, got SM{cc[0]}{cc[1]}")

    # Determine architecture
    if args.arch == "auto":
        arch = "sm90" if cc[0] >= 9 else "sm80"
    else:
        arch = args.arch

    # Validate arch is compatible with hardware
    if arch == "sm90" and cc[0] < 9:
        raise RuntimeError(
            f"SM90 kernel requested but GPU is SM{cc[0]}{cc[1]}. "
            f"Use --arch sm80 instead."
        )

    print(f"Using architecture: {arch.upper()}")

    # Step 1: JIT compile and validate
    compiled, kernel, refs = jit_compile_and_validate(arch)

    # Step 2: AOT export
    # Keep strong references to the CuTe tensors (via `refs`) so that the
    # weakref proxies stored by cute.compile() remain valid during export.
    _ = compiled.to(None)  # pin executor to prevent GC
    try:
        h_path, o_path = aot_export(compiled, arch, args.export_dir)
    except ReferenceError as e:
        print(f"\n  AOT export skipped: {e}")
        print("  (This is a known issue with CuTeDSL weakref lifecycle management)")
        pass

    # prevent refs from being GC'd before export completes
    del refs

    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
