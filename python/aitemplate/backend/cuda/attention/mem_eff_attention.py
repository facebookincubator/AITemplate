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

from ... import registry
from ...backend_spec import CUDASpec

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
// TODO: this include should be removed. There's a bug in CUTLASS, the
// header containing cutlass::gemm::warp::WarpSize is not being included.
// Until the fix is upstreamed, just inject it here instead.
#include "cutlass/gemm/warp/mma.h"
#include "gemm_kernel_utils.h"
#include "kernel_forward.h"

using namespace gemm_kernel_utils;

{{func_signature}}
{

    /*
    problem_sizes0 [b, m, n, k]
    [head_number * batch_size, m, mkv, k0]
    [head_number * batch_size, seq_length, seq_length_kv, head_size]

    problem_sizes1
    [head_number * batch_size, m, k1, mkv]
    [head_number * batch_size, seq_length, head_size_v, seq_length_kv]

    m = seq_len
    n = seq_len
    k = head_size

    Q: B, M, K
    K: B, N, K
    P: B, M, N
    V: B, N, K
    O: B, M, K
    output: bs, num_head, seq_len, head_size
    */


    using ArchTag = cutlass::arch::Sm80;
    constexpr bool kIs64x64 = {{kIs64x64}};
    constexpr bool kSingleValueIteration = {{kSingleValueIteration}};

    // Set grid size
    constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. " \
        "This requires to have `head_size <= kKeysPerBlock` " \
        "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true` (keeps the output in RF rather than GMEM)";
    }

    using GemmType = DefaultGemmType<ArchTag, {{elem_input_type}}>;
    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            {{elem_input_type}},
            {{elem_input_type}},
            {{elem_input_type}}, // ElementC
            float // ElementAccumulator
            >;

    // If the head_size already meets the alignment requirement, then
    // it's safe to mark mem_align to be true to maximize the alignment
    // benefit. Otherwise, assign false to it to use the minimal alignment.
    constexpr const bool mem_align =
        ({{head_size}} % DefaultConfig::kAlignmentA == 0) &&
        ({{head_size}} % DefaultConfig::kAlignmentB == 0);
    using Attention = AttentionKernel<
        {{elem_input_type}}, // scalar_t
        ArchTag,
        mem_align, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    int block_O_size = (*batch_size) * seq_len * num_heads * head_size_v;
    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<{{elem_input_type}}*>(query);
        p.key_ptr = static_cast<{{elem_input_type}}*>(key);
        p.value_ptr = static_cast<{{elem_input_type}}*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = accum_ptr;
        }
        p.output_ptr = static_cast<{{elem_input_type}}*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = seq_len;
        p.num_keys = seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * seq_len;
        p.k_strideH = p.k_strideM * seq_len_kv;
        p.v_strideH = p.v_strideM * seq_len_kv;
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * seq_len * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::string error_msg = std::string("Got error: kernel does not support these inputs") +
           " at " + __FILE__ + ": " + std::to_string(__LINE__);          
      throw std::runtime_error(error_msg);
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
      return;
    }

}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* query,
                   void* key,
                   void* value,
                   float* accum_ptr,
                   int64_t* batch_size,
                   int seq_len,
                   int seq_len_kv,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
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
{{indent}}    {{query}}, {{key}}, {{value}},
{{indent}}    {{accum_ptr}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_len}},
{{indent}}    {{seq_len_kv}},
{{indent}}    {{num_heads}},
{{indent}}    {{head_size}},
{{indent}}    {{head_size_v}},
{{indent}}    {{p_dropout}},
{{indent}}    {{softmax_scale}},
{{indent}}    {{is_causal}}, stream /* default stream */
{{indent}});
    """
)


@registry.reg("cuda.mem_eff_attention.gen_function")
def mem_eff_attention_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        elem_input_type=elem_input_type,
        head_size=func_attrs["head_size"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        kIs64x64="true" if func_attrs["head_size"] <= 64 else "false",
        kSingleValueIteration="true" if func_attrs["head_size"] <= 128 else "false",
    )


@registry.reg("cuda.mem_eff_attention.func_decl")
def mem_eff_attention_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.mem_eff_attention.func_call")
def mem_eff_attention_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 3

    output_name = func_attrs["outputs"][0]._attrs["name"]

    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    batch_size = "&" + xshape[0]._attrs["name"]
    seq_len = x._attrs["shape"][2]._attrs["values"][0]

    num_heads = x._attrs["shape"][1]._attrs["values"][0]
    head_size = x._attrs["shape"][3]._attrs["values"][0]
    p_dropout = func_attrs["dropout"]
    is_causal = func_attrs["causal"]
    softmax_scale = head_size ** (-0.5)

    v = func_attrs["inputs"][2]
    seq_len_kv = v._attrs["shape"][2]._attrs["values"][0]
    head_size_v = v._attrs["shape"][3]._attrs["values"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        accum_ptr="reinterpret_cast<float*>(global_workspace_)",
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        is_causal="true" if is_causal else "false",
        indent=indent,
    )
