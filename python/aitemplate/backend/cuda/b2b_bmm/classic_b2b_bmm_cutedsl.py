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
CuTeDSL backend for classic_b2b_bmm kernel codegen.

Instead of generating CUTLASS C++ template code, this backend:
1. AOT-compiles a CuTeDSL kernel (SM80 or SM90) via cute.compile() + export_to_c()
2. Produces a .h header and .o object file with the embedded cubin
3. Returns a thin C++ wrapper .cu file that #includes the header and delegates
   to the CuTeDSL-generated launch function

The generated artifacts are:
  - <func_name>_cutedsl.h   — CuTeDSL-generated header (Metadata, Load/Unload, wrapper)
  - <func_name>_cutedsl.o   — Object file with embedded cubin(s)
  - <func_name>.cu           — Thin C++ wrapper with the standard AIT function signature

The .cu wrapper maintains the same function signature as the CUTLASS C++ backend:
  void func(void* output, void* query, void* key, void* value, void* bias,
            int64_t batch_size, int64_t num_heads, int64_t m0, int64_t k0,
            cudaStream_t stream)
"""

import functools
import logging
import os
from typing import Any, Dict

import jinja2
from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# C++ wrapper templates
# =============================================================================

# The thin wrapper .cu that bridges AIT's function signature to the
# CuTeDSL-generated launch function.
CUTEDSL_WRAPPER_TEMPLATE = jinja2.Template(
    """
// Auto-generated CuTeDSL wrapper for {{func_name}}
// This file bridges the AIT function interface to the CuTeDSL AOT-compiled kernel.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#include "{{cutedsl_header}}"

namespace {

// Global metadata — loaded once on first call, unloaded at program exit.
static {{func_name}}_cutedsl_Kernel_Metadata_t g_metadata;
static bool g_metadata_loaded = false;

// Ensure cuInit has been called (needed for CUlibrary-based metadata loading).
static void ensure_cu_init() {
    static bool cu_initialized = false;
    if (!cu_initialized) {
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(res, &err_str);
            throw std::runtime_error(
                std::string("cuInit failed: ") + (err_str ? err_str : "unknown error"));
        }
        cu_initialized = true;
    }
}

static void ensure_metadata_loaded() {
    if (!g_metadata_loaded) {
        ensure_cu_init();
        {{func_name}}_cutedsl_Kernel_Metadata_Load(&g_metadata);
        g_metadata_loaded = true;
    }
}

}  // namespace

{{func_signature}} {
    ensure_metadata_loaded();

    // Convert AIT's void* pointers + scalar dims to CuTeDSL typed tensor structs.
    // AIT passes batch_size and num_heads separately; CuTeDSL kernel expects
    // combined batch*heads as the outer dimension of the 3D tensors.
    int32_t batch_heads = static_cast<int32_t>(batch_size * num_heads);
    int32_t m0_i32 = static_cast<int32_t>(m0);
    int32_t k0_i32 = static_cast<int32_t>(k0);

    // Q[batch*heads, M, K]: dynamic_shapes={batch*heads, M}, dynamic_strides={batch_stride}
    // batch_stride = M * K (row-major, elements)
    {{func_name}}_cutedsl_Tensor_mQ_t t_mQ;
    t_mQ.data = query;
    t_mQ.dynamic_shapes[0] = batch_heads;
    t_mQ.dynamic_shapes[1] = m0_i32;
    t_mQ.dynamic_strides[0] = static_cast<int64_t>(m0) * {{n1}};

    // K[batch*heads, N0, K]: dynamic_shapes={batch*heads}
    {{func_name}}_cutedsl_Tensor_mK_t t_mK;
    t_mK.data = key;
    t_mK.dynamic_shapes[0] = batch_heads;

    // V[batch*heads, N0, N1]: dynamic_shapes={batch*heads}
    {{func_name}}_cutedsl_Tensor_mV_t t_mV;
    t_mV.data = value;
    t_mV.dynamic_shapes[0] = batch_heads;

    // Bias[batch, M, N0]: dynamic_shapes={batch_heads, M}, dynamic_strides={batch_stride}
    // Note: bias may be broadcastable; using batch_heads for now
    {{func_name}}_cutedsl_Tensor_mBias_t t_mBias;
    t_mBias.data = bias;
    t_mBias.dynamic_shapes[0] = batch_heads;
    t_mBias.dynamic_shapes[1] = m0_i32;
    t_mBias.dynamic_strides[0] = static_cast<int64_t>(m0) * {{n0}};

    // Out[batch*heads, M, N1]: dynamic_shapes={batch*heads, M}, dynamic_strides={batch_stride}
    {{func_name}}_cutedsl_Tensor_mOut_t t_mOut;
    t_mOut.data = output;
    t_mOut.dynamic_shapes[0] = batch_heads;
    t_mOut.dynamic_shapes[1] = m0_i32;
    t_mOut.dynamic_strides[0] = static_cast<int64_t>(m0) * {{n1}};

    // Convert cudaStream_t to CUstream for the driver API
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    {{func_name}}_cutedsl_wrapper(
        &g_metadata,
        &t_mQ,     // mQ
        &t_mK,     // mK
        &t_mV,     // mV
        &t_mBias,  // mBias
        &t_mOut,   // mOut
        m0_i32,    // m0 (sequence length)
        k0_i32,    // k0
        cu_stream  // CUDA stream
    );
}
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* query,
                   void* key,
                   void* value,
                   void* bias,
                   int64_t batch_size,
                   int64_t num_heads,
                   int64_t m0,
                   int64_t k0,
                   cudaStream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{query}}, {{key}}, {{value}}, {{bias}},
{{indent}}    {{batch_size}},
{{indent}}    {{num_heads}},
{{indent}}    {{m0}},
{{indent}}    {{k0}},
{{indent}}    stream /* default stream */
{{indent}});
    """
)


# =============================================================================
# CuTeDSL AOT compilation (runs at AIT codegen time)
# =============================================================================


@functools.lru_cache(maxsize=32)
def _aot_compile_cutedsl_kernel(
    n0: int,
    n1: int,
    alpha0: float,
    alpha1: float,
    activation_name: str,
    has_causal: bool,
    alpha1_divide_by_seq_len: bool,
    output_dir: str,
    func_name: str,
    arch: int,
):
    """AOT-compile the CuTeDSL b2b_bmm kernel and export .h + .o artifacts.

    This function is cached so repeated calls with the same parameters
    (e.g. during incremental builds) reuse the compilation result.

    Parameters
    ----------
    n0 : int
        Key sequence length (GEMM0 N dimension).
    n1 : int
        Value head dimension (GEMM1 N dimension).
    alpha0, alpha1 : float
        Scale factors for GEMM0 output and after activation.
    activation_name : str
        Activation function name.
    has_causal : bool
        Whether to apply causal masking.
    alpha1_divide_by_seq_len : bool
        Whether to divide alpha1 by sequence length.
    output_dir : str
        Directory to write the .h and .o files.
    func_name : str
        Base name for the exported files.
    arch : int
        GPU architecture (80 for Ampere, 90 for Hopper).

    Returns
    -------
    tuple of (str, str)
        Paths to the generated .h and .o files.
    """
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute
    import torch

    # Select kernel implementation based on arch.
    # NOTE: For now, always use the SM80 kernel (warp-level MMA with cp.async).
    # SM80 warp MMA instructions work on SM90+ hardware as well.
    # The SM90-specific kernel (TMA + WGMMA) requires further debugging of
    # CuTeDSL layout algebra interactions with 3D tensors and will be
    # enabled in a follow-up.
    from aitemplate.backend.cuda.b2b_bmm.cutedsl_b2b_bmm_sm80 import B2bBmmSm80Kernel
    from cutlass.cute.export import export_to_c
    from cutlass.cute.runtime import from_dlpack

    kernel = B2bBmmSm80Kernel(
        n0=n0,
        n1=n1,
        alpha0=alpha0,
        alpha1=alpha1,
        activation_name=activation_name,
        has_causal=has_causal,
        alpha1_divide_by_seq_len=alpha1_divide_by_seq_len,
    )

    # Create representative tensors for compilation.
    # The batch dimension is dynamic (marked via mark_layout_dynamic), so
    # the specific batch size used here doesn't matter.
    rep_batch = 4
    rep_m = 128  # representative M (sequence length)
    rep_k = n1  # K0 = head_dim = N1 (typical for attention)

    q_pt = torch.zeros(rep_batch, rep_m, rep_k, device="cuda", dtype=torch.float16)
    k_pt = torch.zeros(rep_batch, n0, rep_k, device="cuda", dtype=torch.float16)
    v_pt = torch.zeros(rep_batch, n0, n1, device="cuda", dtype=torch.float16)
    bias_pt = torch.zeros(rep_batch, rep_m, n0, device="cuda", dtype=torch.float16)
    out_pt = torch.zeros(rep_batch, rep_m, n1, device="cuda", dtype=torch.float16)

    def make_cute_tensor(t, dynamic_modes_div=None):
        """Create a CuTe tensor with specified modes marked as dynamic.

        For b2b_bmm:
        - Mode 0 (batch) is always dynamic, divisibility=1
        - Mode 1 (M/seq_len) is dynamic for Q, Bias, Out, divisibility=8
        - Inner dims (N0, N1, K) are static compile-time constants

        dynamic_modes_div: list of (mode, divisibility) tuples
        """
        ct = from_dlpack(t, assumed_align=16)
        if dynamic_modes_div is None:
            dynamic_modes_div = [(0, 1)]
        for mode, div in dynamic_modes_div:
            ct = ct.mark_compact_shape_dynamic(
                mode=mode,
                stride_order=t.dim_order(),
                divisibility=div,
            )
        return ct

    # Q[batch, M, K]: batch dynamic (div=1), M dynamic (div=8 for alignment)
    q_cute = make_cute_tensor(q_pt, dynamic_modes_div=[(0, 1), (1, 8)])
    # K[batch, N0, K]: only batch dynamic
    k_cute = make_cute_tensor(k_pt, dynamic_modes_div=[(0, 1)])
    # V[batch, N0, N1]: only batch dynamic
    v_cute = make_cute_tensor(v_pt, dynamic_modes_div=[(0, 1)])
    # Bias[batch, M, N0]: batch and M dynamic
    bias_cute = make_cute_tensor(bias_pt, dynamic_modes_div=[(0, 1), (1, 8)])
    # Out[batch, M, N1]: batch and M dynamic
    out_cute = make_cute_tensor(out_pt, dynamic_modes_div=[(0, 1), (1, 8)])

    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

    _LOGGER.info(f"CuTeDSL: AOT compiling b2b_bmm kernel for {func_name} (SM{arch})")

    # JIT compile (produces MLIR + cubin)
    compiled = cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        bias_cute,
        out_cute,
        rep_m,
        rep_k,
        cu_stream,
    )

    # Export to C artifacts
    os.makedirs(output_dir, exist_ok=True)
    cutedsl_name = f"{func_name}_cutedsl"

    export_to_c(
        compiled,
        file_path=output_dir,
        file_name=cutedsl_name,
    )

    h_path = os.path.join(output_dir, f"{cutedsl_name}.h")
    o_path = os.path.join(output_dir, f"{cutedsl_name}.o")

    assert os.path.exists(h_path), f"CuTeDSL header not generated: {h_path}"
    assert os.path.exists(o_path), f"CuTeDSL object not generated: {o_path}"

    _LOGGER.info(
        f"CuTeDSL: AOT export done — {cutedsl_name}.h ({os.path.getsize(h_path)} bytes), "
        f"{cutedsl_name}.o ({os.path.getsize(o_path)} bytes)"
    )

    return h_path, o_path


# =============================================================================
# AIT backend registry functions
# =============================================================================


@registry.reg("cuda.classic_b2b_bmm.gen_function_cutedsl")
def classic_b2b_bmm_gen_function_cutedsl(func_attrs: Dict[str, Any]) -> str:
    """Generate the CuTeDSL-backed function source for classic_b2b_bmm.

    This AOT-compiles the CuTeDSL kernel, produces .h + .o in the workdir,
    and returns a thin C++ wrapper that delegates to the CuTeDSL launch function.
    """
    q, k, v, bias = func_attrs["inputs"]
    seq_len_dim = 1
    n0 = k._attrs["shape"][seq_len_dim]
    n1 = v._attrs["shape"][-1]
    if not isinstance(n0, IntImm) or not isinstance(n1, IntImm):
        raise RuntimeError(
            f"n0 and n1 must be static dims. {func_attrs['name']=}, {n0=}, {n1=}"
        )

    supported_types = ("float16", "bfloat16")
    input_type = func_attrs["inputs"][0]._attrs["dtype"]
    output_type = func_attrs["outputs"][0]._attrs["dtype"]
    if input_type not in supported_types or output_type not in supported_types:
        raise NotImplementedError(
            f"{supported_types=} for inputs and output "
            f"but got {input_type=} and {output_type=}."
        )

    current_target = Target.current()
    arch = int(current_target._arch)

    has_causal = func_attrs["causal_type"] != CausalType.NO_CAUSAL

    # Get the workdir from func_attrs if available, otherwise use a temp dir
    workdir = func_attrs.get("workdir", "/tmp/ait_cutedsl")
    func_name = func_attrs["name"]

    # AOT compile the CuTeDSL kernel
    h_path, o_path = _aot_compile_cutedsl_kernel(
        n0=n0.value(),
        n1=n1.value(),
        alpha0=float(func_attrs["alpha0"]),
        alpha1=float(func_attrs["alpha1"]),
        activation_name=func_attrs["epilogue_math_name"],
        has_causal=has_causal,
        alpha1_divide_by_seq_len=func_attrs["alpha1_divide_by_seq_len"],
        output_dir=workdir,
        func_name=func_name,
        arch=arch,
    )

    # Store the .o path in func_attrs so the builder can pick it up
    func_attrs["cutedsl_obj_path"] = o_path

    cutedsl_header = f"{func_name}_cutedsl.h"

    # Generate the thin C++ wrapper
    func_signature = FUNC_SIGNATURE.render(func_name=func_name)
    return CUTEDSL_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=func_signature,
        cutedsl_header=cutedsl_header,
        n0=n0.value(),
        n1=n1.value(),
    )


@registry.reg("cuda.classic_b2b_bmm.func_decl_cutedsl")
def classic_b2b_bmm_gen_function_decl_cutedsl(func_attrs: Dict[str, Any]):
    """Generate function declaration (same signature as CUTLASS backend)."""
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.classic_b2b_bmm.func_call_cutedsl")
def classic_b2b_bmm_gen_function_call_cutedsl(func_attrs, indent="  "):
    """Generate a function call (same call site as CUTLASS backend)."""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 4

    output_name = func_attrs["outputs"][0]._attrs["name"]
    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]
    bias_name = func_attrs["inputs"][3]._attrs["name"]

    q_shape = func_attrs["inputs"][0]._attrs["shape"]

    batch_size = q_shape[0]._attrs["name"]
    seq_len_dim = 1
    head_dim = -2
    m0 = q_shape[seq_len_dim]._attrs["name"]

    if len(q_shape) == 3:
        # single head case
        k0 = q_shape[2]._attrs["name"]
        num_heads = "1"
    elif len(q_shape) == 4:
        k0 = q_shape[3]._attrs["name"]
        num_heads = q_shape[head_dim]._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        bias=bias_name,
        batch_size=batch_size,
        num_heads=num_heads,
        m0=m0,
        k0=k0,
        indent=indent,
    )
