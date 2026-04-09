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
CuTeDSL backend for gemm_rcr kernel codegen (no bias).

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
  void func(void* a_ptr, void* b_ptr, void* c_ptr,
            uint8_t* workspace, int split_k,
            int64_t* a_dim0, int64_t* a_dim1,
            int64_t* b_dim0, int64_t* b_dim1,
            int64_t* c_dim0, int64_t* c_dim1,
            cudaStream_t stream)
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

    // Extract M, N, K dimensions from AIT's dim pointers.
    // For gemm_rcr with 2D inputs:
    //   A[M, K] -> a_dim0=M, a_dim1=K
    //   B[N, K] -> b_dim0=N, b_dim1=K
    //   C[M, N] -> c_dim0=M, c_dim1=N
    int32_t M_val = static_cast<int32_t>(*a_dim0);
    int32_t N_val = static_cast<int32_t>(*b_dim0);
    int32_t K_val = static_cast<int32_t>(*a_dim1);

    // A[M, K] row-major: both dims dynamic
    //   dynamic_shapes[0] = M (mode 0), dynamic_shapes[1] = K (mode 1)
    //   dynamic_strides[0] = K (stride of mode 0 = last dim for row-major)
    {{func_name}}_cutedsl_Tensor_mA_t t_mA;
    t_mA.data = a_ptr;
    t_mA.dynamic_shapes[0] = M_val;
    t_mA.dynamic_shapes[1] = K_val;
    t_mA.dynamic_strides[0] = K_val;

    // B[N, K] row-major: both dims dynamic
    {{func_name}}_cutedsl_Tensor_mB_t t_mB;
    t_mB.data = b_ptr;
    t_mB.dynamic_shapes[0] = N_val;
    t_mB.dynamic_shapes[1] = K_val;
    t_mB.dynamic_strides[0] = K_val;

    // C[M, N] row-major: both dims dynamic
    {{func_name}}_cutedsl_Tensor_mC_t t_mC;
    t_mC.data = c_ptr;
    t_mC.dynamic_shapes[0] = M_val;
    t_mC.dynamic_shapes[1] = N_val;
    t_mC.dynamic_strides[0] = N_val;

    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_metadata,
        &t_mA,     // mA [M, K]
        &t_mB,     // mB [N, K]
        &t_mC,     // mC [M, N]
        M_val,     // M dimension
        N_val,     // N dimension
        K_val,     // K dimension
        stream     // CUDA stream
    );
}
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* a_ptr,
                   void* b_ptr,
                   void* c_ptr,
                   uint8_t* workspace,
                   int split_k,
{% for idx in range(input_ndims) %}
                   int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
                   int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
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
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace_,
{{indent}}    {{split_k}},
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
# CuTeDSL AOT compilation (runs at AIT codegen time)
# =============================================================================


@functools.lru_cache(maxsize=32)
def _aot_compile_cutedsl_kernel(
    output_dir: str,
    func_name: str,
    arch: int,
):
    """AOT-compile the CuTeDSL gemm_rcr kernel and export .h + .o artifacts.

    Parameters
    ----------
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
    from cutlass.cute.runtime import from_dlpack

    # Select kernel implementation based on arch
    if arch >= 90:
        from aitemplate.backend.cuda.gemm_universal.cutedsl_gemm_rcr_sm90 import (
            GemmRcrSm90Kernel,
        )

        kernel = GemmRcrSm90Kernel(tile_m=128, tile_n=128)
    else:
        from aitemplate.backend.cuda.gemm_universal.cutedsl_gemm_rcr_sm80 import (
            GemmRcrSm80Kernel,
        )

        kernel = GemmRcrSm80Kernel(tile_m=128, tile_n=128, tile_k=32)

    # Create representative tensors for compilation.
    rep_M = 256
    rep_N = 512
    rep_K = 128

    A_pt = torch.zeros(rep_M, rep_K, device="cuda", dtype=torch.float16)
    B_pt = torch.zeros(rep_N, rep_K, device="cuda", dtype=torch.float16)
    C_pt = torch.zeros(rep_M, rep_N, device="cuda", dtype=torch.float16)

    def make_cute_tensor(t):
        """Create a CuTe tensor with ALL dimensions marked dynamic."""
        ct = from_dlpack(t, assumed_align=16)
        # Mark all modes dynamic so M, N, K can vary at runtime.
        # Mode 0 (M or N): divisibility=1 (no alignment constraint)
        # Mode 1 (K or N): divisibility=8 (128-bit alignment)
        dynamic_modes = [(0, 1), (1, 128 // cutlass.Float16.width)]
        for mode, div in dynamic_modes:
            ct = ct.mark_compact_shape_dynamic(
                mode=mode,
                stride_order=t.dim_order(),
                divisibility=div,
            )
        return ct

    A_cute = make_cute_tensor(A_pt)
    B_cute = make_cute_tensor(B_pt)
    C_cute = make_cute_tensor(C_pt)

    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

    _LOGGER.info(f"CuTeDSL: AOT compiling gemm_rcr kernel for {func_name} (SM{arch})")

    # JIT compile (produces MLIR + cubin)
    compiled = cute.compile(
        kernel,
        A_cute,
        B_cute,
        C_cute,
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
# AIT backend registry functions
# =============================================================================


@registry.reg("cuda.gemm_rcr.gen_function_cutedsl")
def gemm_rcr_gen_function_cutedsl(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    """Generate the CuTeDSL-backed function source for gemm_rcr."""
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

    workdir = func_attrs.get("workdir", "/tmp/ait_cutedsl")
    func_name = func_attrs["name"]

    # AOT compile the CuTeDSL kernel
    h_path, o_path = _aot_compile_cutedsl_kernel(
        output_dir=workdir,
        func_name=func_name,
        arch=arch,
    )

    # Store the .o path so the builder can pick it up
    func_attrs["cutedsl_obj_path"] = o_path

    cutedsl_header = f"{func_name}_cutedsl.h"

    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)

    func_signature = FUNC_SIGNATURE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
    )
    return CUTEDSL_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=func_signature,
        cutedsl_header=cutedsl_header,
    )


@registry.reg("cuda.gemm_rcr.func_decl_cutedsl")
def gemm_rcr_gen_function_decl_cutedsl(func_attrs: Dict[str, Any]):
    """Generate function declaration (same signature as CUTLASS backend)."""
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    func_signature = FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
    ).strip()
    return FUNC_DECL.render(func_signature=func_signature)


@registry.reg("cuda.gemm_rcr.func_call_cutedsl")
def gemm_rcr_gen_function_call_cutedsl(func_attrs, indent="  "):
    """Generate a function call (same call site as CUTLASS backend)."""
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

    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )
