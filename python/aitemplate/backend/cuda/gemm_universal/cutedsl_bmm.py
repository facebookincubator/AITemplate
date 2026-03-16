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
CuTeDSL backend for BMM kernel codegen.

This module provides CuTeDSL-based backends for batched matrix multiply (BMM)
operations with various layouts (CCR, RRR, RCR) and optional residual add.

GPU architecture support:
- SM80 (Ampere): Uses warp-level MMA with cp.async, batch-first tensor ordering.
- SM90 (Hopper): Uses TMA + WGMMA with batch-last tensor ordering (M,K,B).
  Automatically selected when arch >= 90.

Supported operations:
- bmm_ccr / bmm_ccr_add: C[B,M,N] = A[B,K,M]^T @ B[B,N,K]^T (+ D)
- bmm_rrr / bmm_rrr_add: C[B,M,N] = A[B,M,K] @ B[B,K,N] (+ D)
- bmm_rcr / bmm_rcr_add: C[B,M,N] = A[B,M,K] @ B[B,N,K]^T (+ D)
"""

import functools
import logging
import os
from typing import Any, Dict

import jinja2
from aitemplate.backend import registry
from aitemplate.backend.target import Target

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# C++ wrapper templates
# =============================================================================

# The thin wrapper .cu that bridges AIT's function signature to the
# CuTeDSL-generated launch function for BMM without residual add.
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
static {{func_name}}_cutedsl_Kernel_Module_t g_metadata;
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
        {{func_name}}_cutedsl_Kernel_Module_Load(&g_metadata);
        g_metadata_loaded = true;
    }
}

}  // namespace

{{func_signature}} {
    ensure_metadata_loaded();

    // Extract B, M, N, K dimensions from AIT's dim pointers.
    // For bmm with 3D inputs:
    //   A shape depends on layout (CCR: [B,K,M], RRR: [B,M,K])
    //   B shape depends on layout (CCR: [B,N,K], RRR: [B,K,N])
    //   C[B, M, N]
    int32_t B_dim = static_cast<int32_t>(*a_dim0);
    int32_t M_val = static_cast<int32_t>(*{{m_dim_ptr}});
    int32_t N_val = static_cast<int32_t>(*{{n_dim_ptr}});
    int32_t K_val = static_cast<int32_t>(*{{k_dim_ptr}});

    // Setup tensor descriptors for CuTeDSL
    // For 3D BMM tensors, the descriptor format is:
    //   dynamic_shapes[0] = batch size
    //   dynamic_shapes[1] = first inner dimension extent
    //   dynamic_strides[0] = batch stride

    // A tensor
    {{func_name}}_cutedsl_Tensor_mA_t t_mA;
    t_mA.data = a_ptr;
    t_mA.dynamic_shapes[0] = {{a_dynamic_shape_0}};
    t_mA.dynamic_shapes[1] = {{a_dynamic_shape_1}};
    t_mA.dynamic_strides[0] = {{a_dynamic_stride_0}};

    // B tensor
    {{func_name}}_cutedsl_Tensor_mB_t t_mB;
    t_mB.data = b_ptr;
    t_mB.dynamic_shapes[0] = {{b_dynamic_shape_0}};
    t_mB.dynamic_shapes[1] = {{b_dynamic_shape_1}};
    t_mB.dynamic_strides[0] = {{b_dynamic_stride_0}};

    // C tensor (output) [B, M, N]
    {{func_name}}_cutedsl_Tensor_mC_t t_mC;
    t_mC.data = c_ptr;
    t_mC.dynamic_shapes[0] = B_dim;
    t_mC.dynamic_shapes[1] = M_val;
    t_mC.dynamic_strides[0] = M_val * N_val;

    // D tensor (residual or dummy) [B, M, N]
    {{func_name}}_cutedsl_Tensor_mD_t t_mD;
{% if has_d %}
    t_mD.data = d_ptr;
{% else %}
    t_mD.data = c_ptr;  // Dummy pointer, not used
{% endif %}
    t_mD.dynamic_shapes[0] = B_dim;
    t_mD.dynamic_shapes[1] = M_val;
    t_mD.dynamic_strides[0] = M_val * N_val;

    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_metadata,
        &t_mA,     // mA
        &t_mB,     // mB
        &t_mC,     // mC
        &t_mD,     // mD (residual or dummy)
        B_dim,     // Batch dimension
        M_val,     // M dimension
        N_val,     // N dimension
        K_val,     // K dimension
        stream     // CUDA stream
    );
}
"""
)

# The thin wrapper .cu for SM90 (Hopper) with batch-last tensor ordering.
# SM90 tensors: A(M,K,B), B(N,K,B), C(M,N,B), D(M,N,B).
# Each descriptor has 3 dynamic_shapes and 2 dynamic_strides (unit stride is static).
SM90_CUTEDSL_WRAPPER_TEMPLATE = jinja2.Template(
    """
// Auto-generated CuTeDSL SM90 wrapper for {{func_name}}
// Batch-last tensor ordering: A(M,K,B), B(N,K,B), C(M,N,B), D(M,N,B)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#include "{{cutedsl_header}}"

namespace {

static {{func_name}}_cutedsl_Kernel_Module_t g_metadata;
static bool g_metadata_loaded = false;

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
        {{func_name}}_cutedsl_Kernel_Module_Load(&g_metadata);
        g_metadata_loaded = true;
    }
}

}  // namespace

{{func_signature}} {
    ensure_metadata_loaded();

    int32_t B_dim = static_cast<int32_t>(*a_dim0);
    int32_t M_val = static_cast<int32_t>(*{{m_dim_ptr}});
    int32_t N_val = static_cast<int32_t>(*{{n_dim_ptr}});
    int32_t K_val = static_cast<int32_t>(*{{k_dim_ptr}});

    // A tensor: batch-last (M, K, B)
    {{func_name}}_cutedsl_Tensor_mA_t t_mA;
    t_mA.data = a_ptr;
    t_mA.dynamic_shapes[0] = {{a_dynamic_shape_0}};
    t_mA.dynamic_shapes[1] = {{a_dynamic_shape_1}};
    t_mA.dynamic_shapes[2] = {{a_dynamic_shape_2}};
    t_mA.dynamic_strides[0] = {{a_dynamic_stride_0}};
    t_mA.dynamic_strides[1] = {{a_dynamic_stride_1}};

    // B tensor: batch-last (N, K, B)
    {{func_name}}_cutedsl_Tensor_mB_t t_mB;
    t_mB.data = b_ptr;
    t_mB.dynamic_shapes[0] = {{b_dynamic_shape_0}};
    t_mB.dynamic_shapes[1] = {{b_dynamic_shape_1}};
    t_mB.dynamic_shapes[2] = {{b_dynamic_shape_2}};
    t_mB.dynamic_strides[0] = {{b_dynamic_stride_0}};
    t_mB.dynamic_strides[1] = {{b_dynamic_stride_1}};

    // C tensor: batch-last (M, N, B) — strides (N, 1, M*N)
    {{func_name}}_cutedsl_Tensor_mC_t t_mC;
    t_mC.data = c_ptr;
    t_mC.dynamic_shapes[0] = M_val;
    t_mC.dynamic_shapes[1] = N_val;
    t_mC.dynamic_shapes[2] = B_dim;
    t_mC.dynamic_strides[0] = static_cast<int64_t>(N_val);
    t_mC.dynamic_strides[1] = static_cast<int64_t>(M_val) * N_val;

    // D tensor: batch-last (M, N, B) — same layout as C
    {{func_name}}_cutedsl_Tensor_mD_t t_mD;
{% if has_d %}
    t_mD.data = d_ptr;
{% else %}
    t_mD.data = c_ptr;  // Dummy pointer, not used
{% endif %}
    t_mD.dynamic_shapes[0] = M_val;
    t_mD.dynamic_shapes[1] = N_val;
    t_mD.dynamic_shapes[2] = B_dim;
    t_mD.dynamic_strides[0] = static_cast<int64_t>(N_val);
    t_mD.dynamic_strides[1] = static_cast<int64_t>(M_val) * N_val;

    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_metadata,
        &t_mA,
        &t_mB,
        &t_mC,
        &t_mD,
        B_dim,
        M_val,
        N_val,
        K_val,
        stream
    );
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* a_ptr,
                   void* b_ptr,
{% if has_d %}
                   void* d_ptr,
{% endif %}
                   void* c_ptr,
                   uint8_t* workspace,
{% for idx in range(a_ndims) %}
                   int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(b_ndims) %}
                   int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(c_ndims) %}
                   int64_t* c_dim{{idx}},
{% endfor %}
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
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_d %}
{{indent}}    {{d_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace_,
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
    """
)


# =============================================================================
# Layout configurations
# =============================================================================

# Layout-specific parameters for tensor descriptors
#
# IMPORTANT: For 3D BMM tensors, the CuTeDSL tensor descriptor format is:
#   dynamic_shapes[0] = batch size (passed as B_dim from wrapper)
#   dynamic_shapes[1] = first inner dimension extent (M or N)
#   dynamic_strides[0] = batch stride = product of inner dimensions
#
LAYOUT_CONFIGS = {
    # CCR: A[B,K,M] col-major, B[B,N,K] col-major, C[B,M,N] row-major
    # For CCR, A is stored as [B, K, M] so per batch it's [K, M]
    # B is stored as [B, N, K] so per batch it's [N, K]
    "ccr": {
        "a_row_major": False,
        "b_row_major": False,
        "m_dim_ptr": "a_dim2",  # M is last dim of A
        "n_dim_ptr": "b_dim1",  # N is second dim of B
        "k_dim_ptr": "a_dim1",  # K is second dim of A
        # A[B, K, M]: dynamic_shapes={B, K}, dynamic_strides={K*M}
        "a_dynamic_shape_0": "B_dim",
        "a_dynamic_shape_1": "K_val",
        "a_dynamic_stride_0": "K_val * M_val",
        # B[B, N, K]: dynamic_shapes={B, N}, dynamic_strides={N*K}
        "b_dynamic_shape_0": "B_dim",
        "b_dynamic_shape_1": "N_val",
        "b_dynamic_stride_0": "N_val * K_val",
    },
    # RRR: A[B,M,K] row-major, B[B,K,N] row-major, C[B,M,N] row-major
    "rrr": {
        "a_row_major": True,
        "b_row_major": True,
        "m_dim_ptr": "a_dim1",  # M is second dim of A
        "n_dim_ptr": "b_dim2",  # N is last dim of B
        "k_dim_ptr": "a_dim2",  # K is last dim of A
        # A[B, M, K]: dynamic_shapes={B, M}, dynamic_strides={M*K}
        "a_dynamic_shape_0": "B_dim",
        "a_dynamic_shape_1": "M_val",
        "a_dynamic_stride_0": "M_val * K_val",
        # B[B, K, N]: dynamic_shapes={B, K}, dynamic_strides={K*N}
        "b_dynamic_shape_0": "B_dim",
        "b_dynamic_shape_1": "K_val",
        "b_dynamic_stride_0": "K_val * N_val",
    },
    # RCR: A[B,M,K] row-major, B[B,N,K] col-major, C[B,M,N] row-major
    # This is IDEAL for MMA: both A and B have K contiguous, NO TRANSPOSE NEEDED!
    "rcr": {
        "a_row_major": True,
        "b_row_major": False,  # B is col-major [N,K] with K contiguous
        "m_dim_ptr": "a_dim1",  # M is second dim of A
        "n_dim_ptr": "b_dim1",  # N is second dim of B (since B is [B,N,K])
        "k_dim_ptr": "a_dim2",  # K is last dim of A
        # A[B, M, K]: dynamic_shapes={B, M}, dynamic_strides={M*K}
        "a_dynamic_shape_0": "B_dim",
        "a_dynamic_shape_1": "M_val",
        "a_dynamic_stride_0": "M_val * K_val",
        # B[B, N, K]: dynamic_shapes={B, N}, dynamic_strides={N*K}
        "b_dynamic_shape_0": "B_dim",
        "b_dynamic_shape_1": "N_val",
        "b_dynamic_stride_0": "N_val * K_val",
    },
}


# SM90 layout configs — batch-last tensor ordering: A(M,K,B), B(N,K,B), C(M,N,B).
# Each tensor has 3 dynamic_shapes (all modes) and 2 dynamic_strides (non-unit modes).
# The unit-stride mode (stride=1) is static and excluded from dynamic_strides.
#
# Stride derivation from AIT batch-first contiguous tensors permuted to batch-last:
#   RRR A[B,M,K] -> A(M,K,B) strides (K, 1, M*K): mode 1 is unit-stride
#   CCR A[B,K,M] -> A(M,K,B) strides (1, M, K*M): mode 0 is unit-stride
#   RRR B[B,K,N] -> B(N,K,B) strides (1, N, K*N): mode 0 is unit-stride
#   CCR B[B,N,K] -> B(N,K,B) strides (K, 1, N*K): mode 1 is unit-stride
SM90_LAYOUT_CONFIGS = {
    # RRR: A[B,M,K] row-major, B[B,K,N] row-major, C[B,M,N] row-major
    "rrr": {
        "a_row_major": True,
        "b_row_major": True,
        "m_dim_ptr": "a_dim1",
        "n_dim_ptr": "b_dim2",
        "k_dim_ptr": "a_dim2",
        # A(M,K,B) strides (K, 1, M*K): unit-stride at mode 1
        "a_dynamic_shape_0": "M_val",
        "a_dynamic_shape_1": "K_val",
        "a_dynamic_shape_2": "B_dim",
        "a_dynamic_stride_0": "static_cast<int64_t>(K_val)",
        "a_dynamic_stride_1": "static_cast<int64_t>(M_val) * K_val",
        # B(N,K,B) strides (1, N, K*N): unit-stride at mode 0
        "b_dynamic_shape_0": "N_val",
        "b_dynamic_shape_1": "K_val",
        "b_dynamic_shape_2": "B_dim",
        "b_dynamic_stride_0": "static_cast<int64_t>(N_val)",
        "b_dynamic_stride_1": "static_cast<int64_t>(K_val) * N_val",
    },
    # CCR: A[B,K,M] col-major, B[B,N,K] col-major, C[B,M,N] row-major
    "ccr": {
        "a_row_major": False,
        "b_row_major": False,
        "m_dim_ptr": "a_dim2",
        "n_dim_ptr": "b_dim1",
        "k_dim_ptr": "a_dim1",
        # A(M,K,B) strides (1, M, K*M): unit-stride at mode 0
        "a_dynamic_shape_0": "M_val",
        "a_dynamic_shape_1": "K_val",
        "a_dynamic_shape_2": "B_dim",
        "a_dynamic_stride_0": "static_cast<int64_t>(M_val)",
        "a_dynamic_stride_1": "static_cast<int64_t>(K_val) * M_val",
        # B(N,K,B) strides (K, 1, N*K): unit-stride at mode 1
        "b_dynamic_shape_0": "N_val",
        "b_dynamic_shape_1": "K_val",
        "b_dynamic_shape_2": "B_dim",
        "b_dynamic_stride_0": "static_cast<int64_t>(K_val)",
        "b_dynamic_stride_1": "static_cast<int64_t>(N_val) * K_val",
    },
    # RCR: A[B,M,K] row-major, B[B,N,K] col-major, C[B,M,N] row-major
    "rcr": {
        "a_row_major": True,
        "b_row_major": False,
        "m_dim_ptr": "a_dim1",
        "n_dim_ptr": "b_dim1",
        "k_dim_ptr": "a_dim2",
        # A(M,K,B) strides (K, 1, M*K): unit-stride at mode 1 (same as RRR)
        "a_dynamic_shape_0": "M_val",
        "a_dynamic_shape_1": "K_val",
        "a_dynamic_shape_2": "B_dim",
        "a_dynamic_stride_0": "static_cast<int64_t>(K_val)",
        "a_dynamic_stride_1": "static_cast<int64_t>(M_val) * K_val",
        # B(N,K,B) strides (K, 1, N*K): unit-stride at mode 1 (same as CCR B)
        "b_dynamic_shape_0": "N_val",
        "b_dynamic_shape_1": "K_val",
        "b_dynamic_shape_2": "B_dim",
        "b_dynamic_stride_0": "static_cast<int64_t>(K_val)",
        "b_dynamic_stride_1": "static_cast<int64_t>(N_val) * K_val",
    },
}


# =============================================================================
# CuTeDSL AOT compilation
# =============================================================================


@functools.lru_cache(maxsize=32)
def _aot_compile_cutedsl_bmm_kernel(
    output_dir: str,
    func_name: str,
    arch: int,
    a_row_major: bool,
    b_row_major: bool,
    has_d: bool,
):
    """AOT-compile the CuTeDSL BMM kernel and export .h + .o artifacts.

    Parameters
    ----------
    output_dir : str
        Directory to write the .h and .o files.
    func_name : str
        Base name for the exported files.
    arch : int
        GPU architecture (80 for Ampere, 90 for Hopper).
    a_row_major : bool
        True if A is row-major (M,K per batch), False if col-major (K,M).
    b_row_major : bool
        True if B is row-major (K,N per batch), False if col-major (N,K).
    has_d : bool
        If True, enables residual add (C = A @ B + D).

    Returns
    -------
    tuple of (str, str)
        Paths to the generated .h and .o files.
    """
    import cuda.bindings.driver as cuda_drv
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    use_sm90 = arch >= 90

    # Select kernel implementation based on arch.
    if use_sm90:
        from aitemplate.backend.cuda.gemm_universal.cutedsl_bmm_sm90 import (
            BmmSm90Kernel,
        )

        kernel = BmmSm90Kernel(
            tile_m=128,
            tile_n=128,
            a_row_major=a_row_major,
            b_row_major=b_row_major,
            has_d=has_d,
        )
    else:
        from aitemplate.backend.cuda.gemm_universal.cutedsl_bmm_sm80 import (
            BmmSm80Kernel,
        )

        kernel = BmmSm80Kernel(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            a_row_major=a_row_major,
            b_row_major=b_row_major,
            has_d=has_d,
        )

    # Create representative tensors for compilation.
    # All dimensions are dynamic, so specific sizes don't matter.
    # Use sizes divisible by 8 to satisfy alignment requirements.
    rep_B = 8
    rep_M = 256
    rep_N = 512
    rep_K = 128

    # Create batch-first tensors (AIT memory layout)
    if a_row_major:
        A_pt = torch.zeros(rep_B, rep_M, rep_K, device="cuda", dtype=torch.float16)
    else:
        A_pt = torch.zeros(rep_B, rep_K, rep_M, device="cuda", dtype=torch.float16)

    if b_row_major:
        B_pt = torch.zeros(rep_B, rep_K, rep_N, device="cuda", dtype=torch.float16)
    else:
        B_pt = torch.zeros(rep_B, rep_N, rep_K, device="cuda", dtype=torch.float16)

    C_pt = torch.zeros(rep_B, rep_M, rep_N, device="cuda", dtype=torch.float16)
    D_pt = torch.zeros(rep_B, rep_M, rep_N, device="cuda", dtype=torch.float16)

    def make_cute_tensor(t, dynamic_modes_div=None):
        """Create a CuTe tensor with specified modes marked as dynamic.

        dynamic_modes_div: list of (mode, divisibility) tuples
        """
        ct = from_dlpack(t, assumed_align=16)
        if dynamic_modes_div is None:
            dynamic_modes_div = [(0, 1), (1, 8)]
        for mode, div in dynamic_modes_div:
            ct = ct.mark_compact_shape_dynamic(
                mode=mode,
                stride_order=t.dim_order(),
                divisibility=div,
            )
        return ct

    if use_sm90:
        # SM90 kernel expects batch-last tensors: A(M,K,B), B(N,K,B), C(M,N,B).
        # Permute from AIT's batch-first to batch-last (matching test_cutedsl_bmm_sm90.py).
        A_bl = A_pt.permute(1, 2, 0) if a_row_major else A_pt.permute(2, 1, 0)
        B_bl = B_pt.permute(2, 1, 0) if b_row_major else B_pt.permute(1, 2, 0)
        C_bl = C_pt.permute(1, 2, 0)
        D_bl = D_pt.permute(1, 2, 0)

        # Mark all 3 modes dynamic: inner dims (div=8), batch (div=1).
        sm90_modes = [(0, 8), (1, 8), (2, 1)]
        A_cute = make_cute_tensor(A_bl, dynamic_modes_div=sm90_modes)
        B_cute = make_cute_tensor(B_bl, dynamic_modes_div=sm90_modes)
        C_cute = make_cute_tensor(C_bl, dynamic_modes_div=sm90_modes)
        D_cute = make_cute_tensor(D_bl, dynamic_modes_div=sm90_modes)
    else:
        # SM80 batch-first: batch (div=1), first inner dim (div=8).
        sm80_modes = [(0, 1), (1, 8)]
        A_cute = make_cute_tensor(A_pt, dynamic_modes_div=sm80_modes)
        B_cute = make_cute_tensor(B_pt, dynamic_modes_div=sm80_modes)
        C_cute = make_cute_tensor(C_pt, dynamic_modes_div=sm80_modes)
        D_cute = make_cute_tensor(D_pt, dynamic_modes_div=sm80_modes)

    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

    layout_str = ("r" if a_row_major else "c") + ("r" if b_row_major else "c") + "r"
    add_str = "_add" if has_d else ""
    kernel_type = "SM90 TMA+WGMMA" if use_sm90 else "SM80 warp MMA"
    _LOGGER.info(
        f"CuTeDSL: AOT compiling bmm_{layout_str}{add_str} kernel ({kernel_type}) "
        f"for {func_name} (SM{arch})"
    )

    # JIT compile (produces MLIR + cubin)
    compiled = cute.compile(
        kernel,
        A_cute,
        B_cute,
        C_cute,
        D_cute,
        rep_B,
        rep_M,
        rep_N,
        rep_K,
        cu_stream,
    )

    # Export to C artifacts
    os.makedirs(output_dir, exist_ok=True)
    cutedsl_name = f"{func_name}_cutedsl"

    compiled.export_to_c(
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
# Shared generation logic
# =============================================================================


def _gen_function_cutedsl(
    func_attrs: Dict[str, Any],
    layout: str,
    has_d: bool,
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    """Generate the CuTeDSL-backed function source for BMM ops.

    Parameters
    ----------
    func_attrs : dict
        Function attributes from the operator.
    layout : str
        Layout string ("ccr" or "rrr").
    has_d : bool
        Whether this op has a residual D tensor to add.
    """
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
    use_sm90 = arch >= 90

    workdir = func_attrs.get("workdir", "/tmp/ait_cutedsl")
    func_name = func_attrs["name"]

    # Select layout config and wrapper template based on arch
    if use_sm90:
        layout_config = SM90_LAYOUT_CONFIGS[layout]
        wrapper_template = SM90_CUTEDSL_WRAPPER_TEMPLATE
    else:
        layout_config = LAYOUT_CONFIGS[layout]
        wrapper_template = CUTEDSL_WRAPPER_TEMPLATE

    # AOT compile the CuTeDSL kernel
    h_path, o_path = _aot_compile_cutedsl_bmm_kernel(
        output_dir=workdir,
        func_name=func_name,
        arch=arch,
        a_row_major=layout_config["a_row_major"],
        b_row_major=layout_config["b_row_major"],
        has_d=has_d,
    )

    # Store the .o path in func_attrs so the builder can pick it up
    func_attrs["cutedsl_obj_path"] = o_path

    cutedsl_header = f"{func_name}_cutedsl.h"

    a_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    b_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    c_ndims = len(func_attrs["output_accessors"][0].original_shapes)

    # Generate the thin C++ wrapper
    func_signature = FUNC_SIGNATURE.render(
        func_name=func_name,
        has_d=has_d,
        a_ndims=a_ndims,
        b_ndims=b_ndims,
        c_ndims=c_ndims,
    )

    return wrapper_template.render(
        func_name=func_name,
        func_signature=func_signature,
        cutedsl_header=cutedsl_header,
        has_d=has_d,
        **layout_config,
    )


def _gen_function_decl_cutedsl(
    func_attrs: Dict[str, Any],
    has_d: bool,
) -> str:
    """Generate function declaration for CuTeDSL BMM."""
    a_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    b_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    c_ndims = len(func_attrs["output_accessors"][0].original_shapes)

    func_signature = FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        has_d=has_d,
        a_ndims=a_ndims,
        b_ndims=b_ndims,
        c_ndims=c_ndims,
    ).strip()
    return FUNC_DECL.render(func_signature=func_signature)


def _gen_function_call_cutedsl(
    func_attrs: Dict[str, Any],
    has_d: bool,
    indent: str = "  ",
) -> str:
    """Generate a function call for CuTeDSL BMM."""
    from aitemplate.backend.cuda.gemm_universal.common import gen_local_dim_defs

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    c = func_attrs["outputs"][0]

    ashapes = func_attrs["input_accessors"][0].original_shapes
    bshapes = func_attrs["input_accessors"][1].original_shapes
    cshapes = func_attrs["output_accessors"][0].original_shapes

    local_dim_defs = gen_local_dim_defs(func_attrs, indent=indent)
    adims = ["&" + dim._attrs["name"] for dim in ashapes]
    bdims = ["&" + dim._attrs["name"] for dim in bshapes]
    cdims = ["&" + dim._attrs["name"] for dim in cshapes]

    d_ptr = None
    if has_d:
        d_ptr = func_attrs["inputs"][2]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        d_ptr=d_ptr,
        c_ptr=c._attrs["name"],
        has_d=has_d,
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


# =============================================================================
# Registry: bmm_ccr
# =============================================================================


@registry.reg("cuda.bmm_ccr.gen_function_cutedsl")
def _gen_bmm_ccr(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "ccr",
        has_d=False,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_ccr.func_decl_cutedsl")
def _decl_bmm_ccr(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=False)


@registry.reg("cuda.bmm_ccr.func_call_cutedsl")
def _call_bmm_ccr(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=False, indent=indent)


# =============================================================================
# Registry: bmm_rrr
# =============================================================================


@registry.reg("cuda.bmm_rrr.gen_function_cutedsl")
def _gen_bmm_rrr(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "rrr",
        has_d=False,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_rrr.func_decl_cutedsl")
def _decl_bmm_rrr(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=False)


@registry.reg("cuda.bmm_rrr.func_call_cutedsl")
def _call_bmm_rrr(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=False, indent=indent)


# =============================================================================
# Registry: bmm_ccr_add
# =============================================================================


@registry.reg("cuda.bmm_ccr_add.gen_function_cutedsl")
def _gen_bmm_ccr_add(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "ccr",
        has_d=True,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_ccr_add.func_decl_cutedsl")
def _decl_bmm_ccr_add(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=True)


@registry.reg("cuda.bmm_ccr_add.func_call_cutedsl")
def _call_bmm_ccr_add(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=True, indent=indent)


# =============================================================================
# Registry: bmm_rrr_add
# =============================================================================


@registry.reg("cuda.bmm_rrr_add.gen_function_cutedsl")
def _gen_bmm_rrr_add(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "rrr",
        has_d=True,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_rrr_add.func_decl_cutedsl")
def _decl_bmm_rrr_add(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=True)


@registry.reg("cuda.bmm_rrr_add.func_call_cutedsl")
def _call_bmm_rrr_add(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=True, indent=indent)


# =============================================================================
# Registry: bmm_rcr - IDEAL LAYOUT (no transpose needed)
# =============================================================================


@registry.reg("cuda.bmm_rcr.gen_function_cutedsl")
def _gen_bmm_rcr(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "rcr",
        has_d=False,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_rcr.func_decl_cutedsl")
def _decl_bmm_rcr(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=False)


@registry.reg("cuda.bmm_rcr.func_call_cutedsl")
def _call_bmm_rcr(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=False, indent=indent)


# =============================================================================
# Registry: bmm_rcr_add - IDEAL LAYOUT (no transpose needed)
# =============================================================================


@registry.reg("cuda.bmm_rcr_add.gen_function_cutedsl")
def _gen_bmm_rcr_add(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        "rcr",
        has_d=True,
        exec_cond_template=exec_cond_template,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.bmm_rcr_add.func_decl_cutedsl")
def _decl_bmm_rcr_add(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs, has_d=True)


@registry.reg("cuda.bmm_rcr_add.func_call_cutedsl")
def _call_bmm_rcr_add(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, has_d=True, indent=indent)
