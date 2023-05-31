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
attention kernel codegen for ROCM.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec

# pylint: disable=C0301

FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int*>({{name}})")

FUNC_CALL_FP32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<float*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "logging.h"
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"


template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using MaskingSpecialization = ck::tensor_operation::device::MaskingSpecialization;

static constexpr auto MaskingSpec_default = 
    MaskingSpecialization::MaskDisabled;
static constexpr auto MaskingSpec_causal =
    MaskingSpecialization::MaskOutUpperTriangle;

using F32 = float;
using InputType = {{elem_input_type}};

using ADataType        = InputType;
using B0DataType       = InputType;
using B1DataType       = InputType;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = InputType;
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CElementOp    = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

static constexpr auto TensorSpecA  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB0 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB1 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecC  = ck::tensor_operation::device::TensorSpecialization::Default;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGroupedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        ADataType,
        B0DataType,
        B1DataType,
        CDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        B0ElementOp,
        Acc0ElementOp,
        B1ElementOp,
        CElementOp,
        GemmSpec,
        TensorSpecA,
        TensorSpecB0,
        TensorSpecB1,
        TensorSpecC,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        64,          // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        2,           // Gemm1NXdlPerWave
        S<4, 64, 1>, // ABlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<16, 16, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
{% if is_causal %}
        MaskingSpec_causal
{% else %}
        MaskingSpec_default
{% endif %}
    >;   

{{func_signature}}
{
    bool input_permute = true;
    bool output_permute = true;
    
    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{softmax_scale};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    std::vector<typename DeviceGemmInstance::ProblemDesc> problem_descs;

    const char* q_ptr = reinterpret_cast<const char*>(q);
    const char* k_ptr = reinterpret_cast<const char*>(k);
    const char* v_ptr = reinterpret_cast<const char*>(v);
    char* output_ptr = reinterpret_cast<char*>(output);

    std::vector<const void*> q_ptrs;
    std::vector<const void*> k_ptrs;
    std::vector<const void*> v_ptrs;
    std::vector<void*> output_ptrs;

    for(int64_t i = 0; i < batch_size ; i++){
        int M = seqlens[i];
        int N = seqlens[i];
        int K = head_dim;
        int O = head_dim;
        int G0 = 1;
        int G1 = num_heads;

        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
                : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // B0 layout [G0, N, G1, K]
                : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // B1 layout [G0, N, G1, O]
                : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides =
            output_permute
                ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
                : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]

        problem_descs.push_back({a_gs_ms_ks_lengths,
                                 a_gs_ms_ks_strides,
                                 b0_gs_ns_ks_lengths,
                                 b0_gs_ns_ks_strides,
                                 b1_gs_os_ns_lengths,
                                 b1_gs_os_ns_strides,
                                 c_gs_ms_os_lengths,
                                 c_gs_ms_os_strides,
                                 {},   // acc0_biases_gs_ms_ns_lengths
                                 {},   // acc0_biases_gs_ms_ns_strides
                                 {},   // acc1_biases_gs_ms_os_lengths
                                 {}}); // acc1_biases_gs_ms_os_strides

        auto offset = K * G1 * M * sizeof(InputType);
        q_ptrs.push_back(reinterpret_cast<const void*>(q_ptr)); 
        q_ptr += offset;                              
        k_ptrs.push_back(reinterpret_cast<const void*>(k_ptr));   
        k_ptr += offset;                            
        v_ptrs.push_back(reinterpret_cast<const void*>(v_ptr));
        v_ptr += offset;                               
        output_ptrs.push_back(reinterpret_cast<void*>(output_ptr)); 
        output_ptr += offset;                              
    }

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(q_ptrs,
                                      k_ptrs,
                                      v_ptrs,
                                      output_ptrs,
                                      {}, // p_acc0_biases
                                      {}, // p_acc1_biases
                                      problem_descs,
                                      a_element_op,
                                      b0_element_op,
                                      acc0_element_op,
                                      b1_element_op,
                                      c_element_op);

    // specify workspace for problem_desc

    gemm.SetWorkSpacePointer(&argument, workspace);

    if(!gemm.IsSupportedArgument(argument))
    {
        LOG(FATAL) << "wrong! " << gemm.GetTypeString() << " with the specified compilation parameters does not support this Embedding problem.";
    }

    invoker.Run(argument, StreamConfig{stream, false});
}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* q,
                   const void* k,
                   const void* v,
                   const int* seqlens,
                   const int max_seqlen,
                   int64_t batch_size,
                   int num_heads,
                   int head_dim,
                   float softmax_scale,
                   void* workspace,
                   hipStream_t stream)
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
{{indent}}   {{output}}, {{q}}, {{k}}, {{v}}, {{seqlens}},
{{indent}}    {{max_seqlen}}, {{batch_size}},
{{indent}}    {{num_heads}},
{{indent}}    {{head_dim}},
{{indent}}    {{softmax_scale}},
{{indent}}    global_workspace_,
{{indent}}    stream /* default stream */
{{indent}});
    """
)


@registry.reg("rocm.mem_eff_attention.gen_function")
def mem_eff_attention_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    backend_spec = ROCMSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    is_causal = func_attrs["causal"]
    return FUNC_TEMPLATE.render(
        elem_input_type=elem_input_type,
        is_causal=is_causal,
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
    )


@registry.reg("rocm.mem_eff_attention.func_decl")
def mem_eff_attention_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("rocm.mem_eff_attention.func_call")
def mem_eff_attention_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) in [4, 5]

    output_name = func_attrs["outputs"][0]._attrs["name"]

    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]

    seqlens_name = FUNC_CALL_INT32_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][3]._attrs["name"]
    )

    q = func_attrs["inputs"][0]

    batch_size = func_attrs["inputs"][3].shape()[0]._attrs["name"]
    num_heads = q._attrs["shape"][1]._attrs["values"][0]
    max_seqlen = q._attrs["shape"][0].upper_bound() // 16
    head_dim = q._attrs["shape"][3]._attrs["values"][0]

    softmax_scale = head_dim ** (-0.5)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        q=q_name,
        k=k_name,
        v=v_name,
        seqlens=seqlens_name,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        indent=indent,
    )
