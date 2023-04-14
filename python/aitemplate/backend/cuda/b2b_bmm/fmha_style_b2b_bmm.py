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
fmha_style_b2b_bmm kernel codegen for CUDA.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"

#include "fmha_style_b2b_bmm/kernel_forward.h"

namespace {
// Hardcode these sizes for now until we get profiling ready.
constexpr int kQueriesPerBlock = 64;
constexpr int kKeysPerBlock = ({{head_dim_value}} <= 64 ? 64 : 128);
constexpr bool kSingleValueIteration = ({{head_dim_value}} <= kKeysPerBlock);
}  // end namespace

{{func_signature}} {
  using ElementOutput = {{elem_output_type}};
  using ElementAccumulator = {{elem_accum_type}};
  using ElementCompute = {{elem_input_type}};

  using Attention = AttentionKernel<
    ElementCompute,
    ElementAccumulator,
    cutlass::arch::Sm80,  // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kSingleValueIteration,
    {{activation_functor}},
    {{offset_t}}
  >;

  ElementAccumulator alpha0 = ElementAccumulator({{alpha0}});
  ElementAccumulator alpha1 = ElementAccumulator({{alpha1}});

  int64_t head_dim = {{head_dim}};
  int64_t head_dim_value = {{head_dim_value}};

  typename Attention::Params p;
  { // set parameters
    p.query_ptr = static_cast<ElementCompute*>(query);
    p.key_ptr = static_cast<ElementCompute*>(key);
    p.value_ptr = static_cast<ElementCompute*>(value);
    if (bias) {
      p.attn_bias_ptr = static_cast<ElementCompute*>(bias);
    }
    p.output_accum_ptr = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      p.output_accum_ptr = reinterpret_cast<ElementAccumulator*>(accum_ptr);
    }
    p.output_ptr = static_cast<ElementOutput*>(output);

    p.scale = alpha0;
    p.activation_scale = alpha1;
    p.activation_scale_divide_by_seq_len = {{alpha1_divide_by_seq_len}};

    p.num_heads = num_heads;
    p.num_batches = batch_size;

    p.head_dim = head_dim;
    p.head_dim_value = head_dim_value;
    p.seq_length = seq_length;
    p.num_queries = seq_length;
    p.num_keys = seq_length_kv;
    p.causal_type = Attention::Params::{{causal_type}};

    // All tensors are in BMHK shapes
    p.q_strideH = head_dim;
    p.k_strideH = head_dim;
    p.v_strideH = head_dim_value;

    p.q_strideM = p.q_strideH * p.num_heads;
    p.k_strideM = p.k_strideH * p.num_heads;
    p.v_strideM = p.v_strideH * p.num_heads;

    p.q_strideB = p.q_strideM * seq_length;
    p.k_strideB = p.k_strideM * seq_length_kv;
    p.v_strideB = p.v_strideM * seq_length_kv;

    int32_t bias_stride = seq_length_kv;
    {% if bias_broadcast[2] %}
    p.bias_strideM = 0;
    {% else %}
    p.bias_strideM = bias_stride;
    bias_stride *= seq_length;
    {% endif %}

    {% if bias_broadcast[1] %}
    p.bias_strideH = 0;
    {% else %}
    p.bias_strideH = bias_stride;
    bias_stride *= p.num_heads;
    {% endif %}

    {% if bias_broadcast[0] %}
    p.bias_strideB = 0;
    {% else %}
    p.bias_strideB = bias_stride;
    {% endif %}

    p.offset_ptr = static_cast<const {{offset_t}}*>({{offset_ptr}});
  }

  // launch kernel :)
  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    auto result = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    auto error_code = cudaGetLastError();
    if (result != cudaSuccess) {
        throw std::runtime_error(
            "Failed to set attribute! Error: " + std::string(cudaGetErrorString(error_code)) +
            ", error code: " + std::to_string(error_code)
        );
    }
  }
  if (!Attention::check_supported(p)) {
    throw std::runtime_error(
      std::string("Kernel does not support these inputs. ") +
      "Function: {{func_name}}. " +
      "seq_length: " + std::to_string(seq_length) +
      ", head_dim: " + std::to_string({{head_dim}}) +
      ", seq_length_kv: " + std::to_string(seq_length_kv) +
      ", head_dim_value: " + std::to_string({{head_dim_value}}) + "."
    );
  }
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
}
    """
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
  void* output,
  void* query,
  void* key,
  void* value,
  void* bias,
  void* accum_ptr,
  int64_t batch_size,
  int64_t seq_length,
  int64_t seq_length_kv,
  int64_t num_heads,
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
{{indent}}    {{query}},
{{indent}}    {{key}},
{{indent}}    {{value}},
{{indent}}    {{bias}},
{{indent}}    {{accum_ptr}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_length}},
{{indent}}    {{seq_length_kv}},
{{indent}}    {{num_heads}},
{{indent}}    stream
{{indent}});
    """
)


def causal_type_to_kernel_str(causal_type: CausalType) -> str:
    if causal_type == CausalType.NO_CAUSAL:
        return "CausalType::NO_CAUSAL"
    elif causal_type == CausalType.UPPER_RIGHT_EMPTY:
        return "CausalType::UPPER_RIGHT_EMPTY"
    elif causal_type == CausalType.LOWER_LEFT_EMPTY:
        return "CausalType::LOWER_LEFT_EMPTY"
    else:
        raise RuntimeError(f"Unsupported causal type {causal_type=}")


@registry.reg("cuda.fmha_style_b2b_bmm.gen_function")
def fmha_style_b2b_bmm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    q, k, v = func_attrs["inputs"][0:3]

    bias_broadcast = [False] * 4
    if len(func_attrs["inputs"]) > 3:
        bias = func_attrs["inputs"][3]
        bias_broadcast = [var == IntImm(1) for var in bias.shape()]

    k0 = k._attrs["shape"][3]
    n1 = v._attrs["shape"][3]
    if not isinstance(k0, IntImm) or not isinstance(n1, IntImm):
        raise RuntimeError(
            f"k0 and n1 must be static dims. {func_attrs['name']=}, {k0=}, {n1=}"
        )
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_accum_type = elem_input_type
    if elem_input_type == "cutlass:half_t" and not Target.current()._kwargs.get(
        "use_fp16_acc", False
    ):
        elem_accum_type = "float"

    import cutlass_lib

    activation_functor = cutlass_lib.library.EpilogueMathTag[
        cutlass_lib.library.EpilogueMathName[func_attrs["epilogue_math_name"]]
    ]

    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_accum_type=elem_accum_type,
        offset_t="int64_t",
        head_dim=str(k0.value()),
        head_dim_value=str(n1.value()),
        causal_type=causal_type_to_kernel_str(func_attrs["causal_type"]),
        alpha0=str(func_attrs["alpha0"]),
        alpha1=str(func_attrs["alpha1"]),
        alpha1_divide_by_seq_len="true"
        if func_attrs["alpha1_divide_by_seq_len"]
        else "false",
        activation_functor=activation_functor,
        bias_broadcast=bias_broadcast,
        offset_ptr="nullptr",
    )


@registry.reg("cuda.fmha_style_b2b_bmm.func_decl")
def fmha_style_b2b_bmm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.fmha_style_b2b_bmm.func_call")
def fmha_style_b2b_bmm_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    assert len(func_attrs["outputs"]) == 1, f"{len(func_attrs['outputs'])=} != 1"
    assert len(func_attrs["inputs"]) in (
        3,
        4,
    ), f"{len(func_attrs['inputs'])=} != 3 or 4"

    output_name = func_attrs["outputs"][0]._attrs["name"]
    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]

    bias_name = "nullptr"
    if len(func_attrs["inputs"]) == 4:
        bias_name = func_attrs["inputs"][3]._attrs["name"]

    q_shape = func_attrs["inputs"][0]._attrs["shape"]
    k_shape = func_attrs["inputs"][1]._attrs["shape"]
    batch_size = q_shape[0]._attrs["name"]
    seq_length = q_shape[1]._attrs["name"]
    seq_length_kv = k_shape[1]._attrs["name"]
    num_heads = q_shape[2]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        bias=bias_name,
        accum_ptr="global_workspace_",
        batch_size=batch_size,
        seq_length=seq_length,
        seq_length_kv=seq_length_kv,
        num_heads=num_heads,
        indent=indent,
    )
