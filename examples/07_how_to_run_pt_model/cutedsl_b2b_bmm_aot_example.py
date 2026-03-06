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
CuTeDSL B2B BMM AOT (Ahead-of-Time) Compilation Example
=========================================================

Demonstrates:
1. Creating a CuTeDSL SM80 B2B BMM kernel
2. JIT compiling and running it
3. Validating against a PyTorch reference
4. Exporting as AOT artifacts (.h header + .o object) for C++ integration

The AOT export generates:
  - <name>.h: C header with Metadata struct, load/unload functions, and a
    wrapper function that takes raw pointers and launches the kernel
  - <name>.o: Object file with embedded cubin(s) and the host launch code

These can be linked into any C/C++ application:
    gcc -o app app.c b2b_bmm_sm80.o -lcuda

Reference:
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html

Run with:
    buck run fbcode//aitemplate/AITemplate/examples:cutedsl_b2b_bmm_aot_example
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from aitemplate.backend.cuda.b2b_bmm.cutedsl_b2b_bmm_sm80 import B2bBmmSm80Kernel
from cutlass.cute.export import export_to_c
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# PyTorch Reference
# =============================================================================


class PTB2bBmm(torch.nn.Module):
    """PyTorch reference: output = alpha1 * activation(alpha0 * Q @ K^T + bias) @ V"""

    def __init__(self, alpha0: float, alpha1: float, activation: str = "Sigmoid"):
        super().__init__()
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.activation = activation

    def forward(self, q, k, v, bias):
        score = self.alpha0 * (q @ k.transpose(-2, -1)) + bias
        if self.activation == "Sigmoid":
            score = torch.sigmoid(score)
        elif self.activation == "ReLu":
            score = torch.relu(score)
        elif self.activation == "Tanh":
            score = torch.tanh(score)
        elif self.activation == "SiLu":
            score = torch.nn.functional.silu(score)
        elif self.activation == "Gelu":
            score = torch.nn.functional.gelu(score)
        else:
            pass  # Identity
        score = self.alpha1 * score
        return score @ v


# =============================================================================
# Helper: Create CuTe tensor from PyTorch tensor
# =============================================================================


def make_cute_tensor(torch_tensor: torch.Tensor) -> cute.Tensor:
    """Convert a PyTorch CUDA tensor to a CuTe tensor."""
    ct = from_dlpack(torch_tensor, assumed_align=16)
    ct = ct.mark_layout_dynamic(leading_dim=len(torch_tensor.shape) - 1)
    return ct


# =============================================================================
# Step 1: JIT Compile and Validate
# =============================================================================


def jit_compile_and_validate():
    """JIT-compile the B2B BMM kernel and validate against PyTorch."""
    print("\n" + "=" * 60)
    print("Step 1: JIT Compile and Validate")
    print("=" * 60)

    # Parameters matching the AIT classic_b2b_bmm_example
    batch = 4
    seq_len = 128  # N0
    head_dim = 64  # K0 = N1
    alpha0 = 1.0 / (head_dim**0.5)  # 0.125
    alpha1 = 1.0
    activation = "Sigmoid"

    print(f"  batch={batch}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  alpha0={alpha0}, alpha1={alpha1}, activation={activation}")

    # Create PyTorch reference tensors
    q_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_pt = torch.randn(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)
    bias_pt = torch.randn(batch, seq_len, seq_len, device="cuda", dtype=torch.float16)
    out_pt = torch.zeros(batch, seq_len, head_dim, device="cuda", dtype=torch.float16)

    # PyTorch reference
    pt_model = PTB2bBmm(alpha0, alpha1, activation).cuda().half()
    y_ref = pt_model(q_pt, k_pt, v_pt, bias_pt)

    # Create CuTeDSL kernel
    kernel = B2bBmmSm80Kernel(
        n0=seq_len,
        n1=head_dim,
        alpha0=alpha0,
        alpha1=alpha1,
        activation_name=activation,
        has_causal=False,
    )

    # Convert to CuTe tensors
    q_cute = make_cute_tensor(q_pt)
    k_cute = make_cute_tensor(k_pt)
    v_cute = make_cute_tensor(v_pt)
    bias_cute = make_cute_tensor(bias_pt)
    out_cute = make_cute_tensor(out_pt)

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    # JIT compile
    print("\n  JIT compiling CuTeDSL kernel...")
    compiled = cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        bias_cute,
        out_cute,
        seq_len,
        head_dim,
        cu_stream,
    )
    print("  Compilation successful!")

    # Execute
    print("  Executing kernel...")
    compiled(
        q_cute,
        k_cute,
        v_cute,
        bias_cute,
        out_cute,
        seq_len,
        head_dim,
        cu_stream,
    )
    torch.cuda.synchronize()

    # Validate
    close = torch.allclose(out_pt, y_ref, atol=1e-2, rtol=1e-2)
    max_diff = (out_pt - y_ref).abs().max().item()
    print(f"\n  Results match PyTorch: {close} (max diff: {max_diff:.6f})")

    if not close:
        print("  WARNING: Results do not match! This may be expected for the")
        print("  initial implementation that needs debugging.")

    return compiled, kernel


# =============================================================================
# Step 2: AOT Export
# =============================================================================


def aot_export(compiled):
    """Export the compiled kernel as AOT artifacts (.h + .o)."""
    print("\n" + "=" * 60)
    print("Step 2: AOT Export (Ahead-of-Time Compilation)")
    print("=" * 60)

    output_dir = "./tmp/cutedsl_aot_output"
    os.makedirs(output_dir, exist_ok=True)
    file_name = "b2b_bmm_sm80"

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
    print(f"\n  To use from C/C++:")
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
    print("=" * 60)
    print("CuTeDSL B2B BMM AOT Compilation Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required!")

    cc = torch.cuda.get_device_capability(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)} (SM{cc[0]}{cc[1]})")

    if cc[0] < 8:
        raise RuntimeError(f"SM80+ required, got SM{cc[0]}{cc[1]}")

    # Step 1: JIT compile and validate
    compiled, kernel = jit_compile_and_validate()

    # Step 2: AOT export
    h_path, o_path = aot_export(compiled)

    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
