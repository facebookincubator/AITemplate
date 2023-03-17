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
attention kernel codegen for CUDA.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry

# pylint: disable=C0301

FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int*>({{name}})")

FUNC_CALL_FP32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<float*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "fmha.h"
#include "fmha_fprop_kernel_1xN.h"

namespace {

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax>
__global__ void fmha_fprop_fp16_sm80_loop_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax>(params);
}

template<typename Kernel_traits>
void run_fmha_fp16_sm80_loop_(Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
                            const bool configure) {
    bool is_causal = launch_params.params.is_causal;
    auto kernel = (is_causal
           ? (&fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, false, true, false>)
           : (&fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, false, false, false>));

    constexpr int N = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.s + N - 1) / N;
    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    if (configure) {
        using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr int M = Kernel_traits::Cta_tile_p::M;
        size_t STEPS = (launch_params.params.s + M - 1) / M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8 * loop_steps;
        launch_params.elts_per_thread = elts_per_head;
        return;
    }

    dim3 grid(launch_params.params.h, launch_params.params.b);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_fp16_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
                        const bool configure) {
{{custom_kernel}}
}

void set_params(Fused_multihead_attention_fprop_params &params,
                // sizes
                const size_t b,
                const size_t s,
                const size_t h,
                const size_t d,
                // device pointers
                void *qkv_packed_d,
                void *cu_seqlens_d,
                void *o_packed_d,
                void *o_tmp_d,
                void *do_packed_d,
                void *s_d,
                void *softmax_lse_d,
                void *dsoftmax_sum_d,
                float p_dropout,
                float softmax_scale,
                bool is_causal) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = qkv_packed_d;
    params.k_ptr = qkv_packed_d + get_size_in_bytes(h * d, data_type);
    params.v_ptr = qkv_packed_d + 2 * get_size_in_bytes(h * d, data_type);
    params.q_row_stride_in_elts = 3 * h * d;
    params.k_row_stride_in_elts = 3 * h * d;
    params.v_row_stride_in_elts = 3 * h * d;
    params.q_head_stride_in_elts = d;
    params.k_head_stride_in_elts = d;
    params.v_head_stride_in_elts = d;
    params.o_ptr = o_packed_d;
    params.o_row_stride_in_elts = h * d;
    params.o_head_stride_in_elts = d;
    params.do_ptr = do_packed_d;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;
    params.dsoftmax_sum = dsoftmax_sum_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}
}  // namespace

{{func_signature}}
{
    bool is_dropout = p_dropout > 0.0;
    bool return_softmax = false;

    Launch_params<Fused_multihead_attention_fprop_params> launch_params(stream, is_dropout, return_softmax);

    set_params(launch_params.params,
               batch_size, // b
               seq_len, // s
               num_heads, // h
               head_size, // d
               (void*)qkv,
               (void*)cu_seqlens,
               (void*)output,
               loop ? (void*)o_tmp : nullptr,
               nullptr,
               nullptr, // return softmax
               (void*)softmax_lse,
               nullptr,
               p_dropout,
               softmax_scale,
               is_causal);

    run_fmha_fp16_sm80(launch_params, /*configure=*/ false);
}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* qkv,
                   const int* cu_seqlens,
                   float* softmax_lse,
                   float* o_tmp,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_size,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   bool loop,
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
{{indent}}   {{output}}, {{qkv}}, {{cu_seqlens}},
{{indent}}    {{softmax_lse}}, {{o_tmp}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_len}},
{{indent}}    {{num_heads}},
{{indent}}    {{head_size}},
{{indent}}    {{p_dropout}},
{{indent}}    {{softmax_scale}},
{{indent}}    {{is_causal}}, {{loop}}, stream /* default stream */
{{indent}});
    """
)

ATT_KERNEL_TEMPLATE = jinja2.Template(
    """
    using Kernel_traits = FMHA_kernel_traits<{{s1}}, {{s2}}, 16, 1, 4, 0x08u>;
    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
    """
)


@registry.reg("cuda.flash_attention.gen_function")
def flash_attention_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    return FUNC_TEMPLATE.render(
        custom_kernel=ATT_KERNEL_TEMPLATE.render(
            s1=128 if func_attrs["seq_len"] == 128 else 256,
            s2=func_attrs["head_size"],
        ),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
    )


@registry.reg("cuda.flash_attention.func_decl")
def flash_attention_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.flash_attention.func_call")
def flash_attention_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 2

    output_name = func_attrs["outputs"][0]._attrs["name"]

    qkv_name = func_attrs["inputs"][0]._attrs["name"]

    seqlens_name = FUNC_CALL_INT32_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )

    x = func_attrs["inputs"][0]

    batch_size = func_attrs["batch_size"]
    seq_len = func_attrs["seq_len"]

    num_heads = x._attrs["shape"][2]._attrs["values"][0]
    head_size = x._attrs["shape"][3]._attrs["values"][0]
    p_dropout = func_attrs["dropout"]
    is_causal = func_attrs["causal"]
    softmax_scale = head_size ** (-0.5)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        qkv=qkv_name,
        cu_seqlens=seqlens_name,
        softmax_lse="reinterpret_cast<float*>(global_workspace_)",
        o_tmp="reinterpret_cast<float*>(global_workspace_ + {} * sizeof(float))".format(
            batch_size * num_heads * seq_len
        ),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_size=head_size,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        is_causal="true" if is_causal else "false",
        loop="true" if seq_len > 256 else "false",
        indent=indent,
    )
