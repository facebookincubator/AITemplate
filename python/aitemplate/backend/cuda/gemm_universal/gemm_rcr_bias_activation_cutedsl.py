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
CuTeDSL backend for gemm_rcr_bias with fused epilogue activations.

Supports the following fused ops (all share the same 3-input signature):
  - gemm_rcr_bias_relu:    ReLU(A @ B^T + Bias)
  - gemm_rcr_bias_sigmoid: Sigmoid(A @ B^T + Bias)
  - gemm_rcr_bias_swish:   SiLU(A @ B^T + Bias)  = x * sigmoid(x)

Strategy:
  The activation is fused directly into the CuTeDSL kernel epilogue via
  a compile-time ``fusion_type`` constant.  Each fusion_type value produces
  a separately AOT-compiled kernel with the activation baked in -- there
  is zero runtime branching overhead.

  Fusion types (matches GemmRcrBias{Sm80,Sm90}Kernel constants):
    0: identity (plain gemm_rcr_bias)
    1: relu
    2: sigmoid
    3: swish / SiLU

Supported broadcast (extra-input) ops:
  - gemm_rcr_bias_sigmoid_mul: Sigmoid(A @ B^T + Bias) * D0
  - gemm_rcr_bias_mul_add:    (A @ B^T + Bias) * D0 + D1

  For broadcast ops the base GEMM+bias+activation kernel runs first, then
  a lightweight element-wise CUDA kernel applies the D0/D1 post-processing
  in a second launch.  The D0/D1 tensors are not part of the CuTeDSL kernel
  signature so they must be handled externally.
"""

import logging
from typing import Any, Dict

import jinja2
from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal.gemm_rcr_bias_cutedsl import (
    _aot_compile_cutedsl_kernel,
    CUTEDSL_WRAPPER_TEMPLATE,
    FUNC_CALL_TEMPLATE as BIAS_FUNC_CALL_TEMPLATE,
    FUNC_DECL,
    FUNC_SIGNATURE as BIAS_FUNC_SIGNATURE,
)
from aitemplate.backend.target import Target

_LOGGER = logging.getLogger(__name__)


# Mapping from op name suffix to fusion_type integer
_FUSION_TYPES = {
    "relu": 1,
    "sigmoid": 2,
    "swish": 3,
}


# =============================================================================
# Broadcast wrapper template (sigmoid_mul, mul_add)
# =============================================================================

BROADCAST_FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* a_ptr,
                   void* b_ptr,
                   void* bias_ptr,
                   void* d0_ptr,
{% if has_d1 %}
                   void* d1_ptr,
{% endif %}
                   void* c_ptr,
                   uint8_t* workspace,
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

BROADCAST_FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{bias_ptr}},
{{indent}}    {{d0_ptr}},
{% if has_d1 %}
{{indent}}    {{d1_ptr}},
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

CUTEDSL_BROADCAST_WRAPPER_TEMPLATE = jinja2.Template(
    """
// Auto-generated CuTeDSL wrapper for {{func_name}}
// Computes: {{operation_desc}}

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// Element-wise post-processing kernel for D0 (and D1).
__global__ void {{func_name}}_post_kernel(
    half* c_ptr,
    const half* d0_ptr,
{% if has_d1 %}
    const half* d1_ptr,
{% endif %}
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        half val = c_ptr[idx];
        {{post_expr}}
        c_ptr[idx] = val;
    }
}

}  // namespace

{{func_signature}} {
    ensure_metadata_loaded();

    int32_t M_val = static_cast<int32_t>(*a_dim0);
    int32_t N_val = static_cast<int32_t>(*b_dim0);
    int32_t K_val = static_cast<int32_t>(*a_dim1);

    {{func_name}}_cutedsl_Tensor_mA_t t_mA;
    t_mA.data = a_ptr;
    t_mA.dynamic_shapes[0] = K_val;
    t_mA.dynamic_strides[0] = K_val;

    {{func_name}}_cutedsl_Tensor_mB_t t_mB;
    t_mB.data = b_ptr;
    t_mB.dynamic_shapes[0] = K_val;
    t_mB.dynamic_strides[0] = K_val;

    {{func_name}}_cutedsl_Tensor_mBias_t t_mBias;
    t_mBias.data = bias_ptr;
    t_mBias.dynamic_shapes[0] = N_val;

    {{func_name}}_cutedsl_Tensor_mC_t t_mC;
    t_mC.data = c_ptr;
    t_mC.dynamic_shapes[0] = N_val;
    t_mC.dynamic_strides[0] = N_val;

    // Step 1: GEMM + bias + activation (fused in CuTeDSL kernel)
    cute_dsl_{{func_name}}_cutedsl_wrapper(
        &g_metadata,
        &t_mA, &t_mB, &t_mBias, &t_mC,
        M_val, N_val, K_val, stream
    );

    // Step 2: Apply element-wise D0 (and D1) post-processing
    int64_t num_elements = static_cast<int64_t>(M_val) * N_val;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    {{func_name}}_post_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<half*>(c_ptr),
        static_cast<const half*>(d0_ptr),
{% if has_d1 %}
        static_cast<const half*>(d1_ptr),
{% endif %}
        num_elements
    );
}
"""
)


# =============================================================================
# Shared helpers for 3-input activation ops (relu, sigmoid, swish)
# =============================================================================


def _gen_function_cutedsl(
    func_attrs: Dict[str, Any],
    fusion_type: int,
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    """Generate CuTeDSL-backed function source with fused epilogue activation."""
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

    # AOT compile with the fused epilogue
    h_path, o_path = _aot_compile_cutedsl_kernel(
        output_dir=workdir,
        func_name=func_name,
        arch=arch,
        fusion_type=fusion_type,
    )

    func_attrs["cutedsl_obj_path"] = o_path
    cutedsl_header = f"{func_name}_cutedsl.h"

    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)

    func_signature = BIAS_FUNC_SIGNATURE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
    )
    # Reuse the plain gemm_rcr_bias wrapper template -- the activation is
    # already fused inside the CuTeDSL kernel, so no extra CUDA kernel needed.
    return CUTEDSL_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=func_signature,
        cutedsl_header=cutedsl_header,
    )


def _gen_function_decl_cutedsl(func_attrs: Dict[str, Any]):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    func_signature = BIAS_FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
    ).strip()
    return FUNC_DECL.render(func_signature=func_signature)


def _gen_function_call_cutedsl(func_attrs, indent="  "):
    from aitemplate.backend.cuda.gemm_universal.common import gen_local_dim_defs

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    bias = func_attrs["inputs"][2]
    c = func_attrs["outputs"][0]

    ashapes = func_attrs["input_accessors"][0].original_shapes
    bshapes = func_attrs["input_accessors"][1].original_shapes
    cshapes = func_attrs["output_accessors"][0].original_shapes

    local_dim_defs = gen_local_dim_defs(func_attrs, indent=indent)
    adims = ["&" + dim._attrs["name"] for dim in ashapes]
    bdims = ["&" + dim._attrs["name"] for dim in bshapes]
    cdims = ["&" + dim._attrs["name"] for dim in cshapes]

    return BIAS_FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        bias_ptr=bias._attrs["name"],
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


# =============================================================================
# Broadcast helpers (sigmoid_mul, mul_add)
# =============================================================================


_BROADCAST_POST_EXPRS = {
    "sigmoid_mul": (
        "val = __hmul(val, d0_ptr[idx]);",
        False,
        "Sigmoid(A @ B^T + Bias) * D0",
        _FUSION_TYPES["sigmoid"],  # fuse sigmoid into kernel
    ),
    "mul_add": (
        "val = __hadd(__hmul(val, d0_ptr[idx]), d1_ptr[idx]);",
        True,
        "(A @ B^T + Bias) * D0 + D1",
        0,  # no activation fused, just bias
    ),
}


def _gen_broadcast_function_cutedsl(
    func_attrs: Dict[str, Any],
    op_name: str,
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
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

    post_expr, has_d1, operation_desc, fusion_type = _BROADCAST_POST_EXPRS[op_name]

    # AOT compile with fused activation (e.g., sigmoid for sigmoid_mul)
    h_path, o_path = _aot_compile_cutedsl_kernel(
        output_dir=workdir,
        func_name=func_name,
        arch=arch,
        fusion_type=fusion_type,
    )
    func_attrs["cutedsl_obj_path"] = o_path
    cutedsl_header = f"{func_name}_cutedsl.h"

    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)

    func_signature = BROADCAST_FUNC_SIGNATURE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        has_d1=has_d1,
    )
    return CUTEDSL_BROADCAST_WRAPPER_TEMPLATE.render(
        func_name=func_name,
        func_signature=func_signature,
        cutedsl_header=cutedsl_header,
        post_expr=post_expr,
        has_d1=has_d1,
        operation_desc=operation_desc,
    )


def _gen_broadcast_function_decl_cutedsl(func_attrs, has_d1):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    func_signature = BROADCAST_FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        has_d1=has_d1,
    ).strip()
    return FUNC_DECL.render(func_signature=func_signature)


def _gen_broadcast_function_call_cutedsl(func_attrs, has_d1, indent="  "):
    from aitemplate.backend.cuda.gemm_universal.common import gen_local_dim_defs

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    bias = func_attrs["inputs"][2]
    d0 = func_attrs["inputs"][3]
    d1 = func_attrs["inputs"][4] if has_d1 else None
    c = func_attrs["outputs"][0]

    ashapes = func_attrs["input_accessors"][0].original_shapes
    bshapes = func_attrs["input_accessors"][1].original_shapes
    cshapes = func_attrs["output_accessors"][0].original_shapes

    local_dim_defs = gen_local_dim_defs(func_attrs, indent=indent)
    adims = ["&" + dim._attrs["name"] for dim in ashapes]
    bdims = ["&" + dim._attrs["name"] for dim in bshapes]
    cdims = ["&" + dim._attrs["name"] for dim in cshapes]

    return BROADCAST_FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        bias_ptr=bias._attrs["name"],
        d0_ptr=d0._attrs["name"],
        d1_ptr=d1._attrs["name"] if d1 else "",
        c_ptr=c._attrs["name"],
        has_d1=has_d1,
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


# =============================================================================
# Registry: gemm_rcr_bias_relu
# =============================================================================


@registry.reg("cuda.gemm_rcr_bias_relu.gen_function_cutedsl")
def _gen_relu(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        _FUSION_TYPES["relu"],
        exec_cond_template,
        dim_info_dict,
    )


@registry.reg("cuda.gemm_rcr_bias_relu.func_decl_cutedsl")
def _decl_relu(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_relu.func_call_cutedsl")
def _call_relu(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, indent)


# =============================================================================
# Registry: gemm_rcr_bias_sigmoid
# =============================================================================


@registry.reg("cuda.gemm_rcr_bias_sigmoid.gen_function_cutedsl")
def _gen_sigmoid(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        _FUSION_TYPES["sigmoid"],
        exec_cond_template,
        dim_info_dict,
    )


@registry.reg("cuda.gemm_rcr_bias_sigmoid.func_decl_cutedsl")
def _decl_sigmoid(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_sigmoid.func_call_cutedsl")
def _call_sigmoid(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, indent)


# =============================================================================
# Registry: gemm_rcr_bias_swish (SiLU)
# =============================================================================


@registry.reg("cuda.gemm_rcr_bias_swish.gen_function_cutedsl")
def _gen_swish(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_function_cutedsl(
        func_attrs,
        _FUSION_TYPES["swish"],
        exec_cond_template,
        dim_info_dict,
    )


@registry.reg("cuda.gemm_rcr_bias_swish.func_decl_cutedsl")
def _decl_swish(func_attrs):
    return _gen_function_decl_cutedsl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_swish.func_call_cutedsl")
def _call_swish(func_attrs, indent="  "):
    return _gen_function_call_cutedsl(func_attrs, indent)


# =============================================================================
# Registry: gemm_rcr_bias_sigmoid_mul
# =============================================================================


@registry.reg("cuda.gemm_rcr_bias_sigmoid_mul.gen_function_cutedsl")
def _gen_sigmoid_mul(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_broadcast_function_cutedsl(
        func_attrs, "sigmoid_mul", exec_cond_template, dim_info_dict
    )


@registry.reg("cuda.gemm_rcr_bias_sigmoid_mul.func_decl_cutedsl")
def _decl_sigmoid_mul(func_attrs):
    return _gen_broadcast_function_decl_cutedsl(func_attrs, has_d1=False)


@registry.reg("cuda.gemm_rcr_bias_sigmoid_mul.func_call_cutedsl")
def _call_sigmoid_mul(func_attrs, indent="  "):
    return _gen_broadcast_function_call_cutedsl(func_attrs, has_d1=False, indent=indent)


# =============================================================================
# Registry: gemm_rcr_bias_mul_add
# =============================================================================


@registry.reg("cuda.gemm_rcr_bias_mul_add.gen_function_cutedsl")
def _gen_mul_add(func_attrs, exec_cond_template=None, dim_info_dict=None):
    return _gen_broadcast_function_cutedsl(
        func_attrs, "mul_add", exec_cond_template, dim_info_dict
    )


@registry.reg("cuda.gemm_rcr_bias_mul_add.func_decl_cutedsl")
def _decl_mul_add(func_attrs):
    return _gen_broadcast_function_decl_cutedsl(func_attrs, has_d1=True)


@registry.reg("cuda.gemm_rcr_bias_mul_add.func_call_cutedsl")
def _call_mul_add(func_attrs, indent="  "):
    return _gen_broadcast_function_call_cutedsl(func_attrs, has_d1=True, indent=indent)
