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
grouped_fmha_style_b2b_bmm kernel codegen for CUDA.
"""
from typing import Any, Dict

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.b2b_bmm import fmha_style_b2b_bmm
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm

from ... import registry

# pylint: disable=C0301

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
  void* output,
  void* query,
  void* key,
  void* value,
  void* bias,

  // Used as an internal cache to compute output values when the output is too
  // large to be computed in a single iteration.
  void* accum_ptr,

  int64_t batch_size,

  // Max sequence lengths of the query, key and values.
  // This kernel always assumes that seq_length == seq_length_kv.
  int64_t seq_length,
  int64_t seq_length_kv,

  int64_t num_heads,

  // A pointer to the offset of the variable sequence lengths
  // of the query and key tensors.
  // e.g. when batch_size=4, seq_length is [2, 1, 4, 5]
  // offset array is [0, 2, 3, 7, 12].
  const void* offset,

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
{{indent}}    {{accum_ptr}},
{{indent}}    {{batch_size}},
{{indent}}    {{seq_length}},
{{indent}}    {{seq_length_kv}},
{{indent}}    {{num_heads}},
{{indent}}    {{offset}},
{{indent}}    stream
{{indent}});
    """
)


@registry.reg("cuda.grouped_fmha_style_b2b_bmm.gen_function")
def grouped_fmha_style_b2b_bmm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """the function for generating attention kernel"""
    q, k, v = func_attrs["inputs"][0:3]

    bias_broadcast = [False] * 4
    if len(func_attrs["inputs"]) > 3:
        bias = func_attrs["inputs"][3]
        bias_broadcast = [var == IntImm(1) for var in bias.shape()]

    jagged_dim = q._attrs["shape"][0]
    head_dim = q._attrs["shape"][2]
    head_dim_value = v._attrs["shape"][2]
    if not isinstance(head_dim, IntImm) or not isinstance(head_dim_value, IntImm):
        raise RuntimeError(
            f"head_dim and head_dim_value must be static dims. {func_attrs['name']=}, {head_dim=}, {head_dim_value=}"
        )
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_accum_type = elem_input_type
    if (
        elem_input_type == "cutlass::half_t"
        and "use_fp16_acc" in Target.current()._kwargs
        and not Target.current()._kwargs["use_fp16_acc"]
    ):
        elem_accum_type = "float"

    import cutlass_lib

    activation_functor = cutlass_lib.library.EpilogueMathTag[
        cutlass_lib.library.EpilogueMathName[func_attrs["epilogue_math_name"]]
    ]
    return fmha_style_b2b_bmm.FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_accum_type=elem_accum_type,
        offset_t=jagged_dim.offsets_type(),
        seq_length="max_seq_length",
        seq_length_kv="max_seq_length",
        head_dim=str(head_dim.value()),
        head_dim_value=str(head_dim_value.value()),
        causal_type=fmha_style_b2b_bmm.causal_type_to_kernel_str(
            func_attrs["causal_type"]
        ),
        num_heads="num_heads",
        alpha0=str(func_attrs["alpha0"]),
        alpha1=str(func_attrs["alpha1"]),
        alpha1_divide_by_seq_len="true"
        if func_attrs["alpha1_divide_by_seq_len"]
        else "false",
        activation_functor=activation_functor,
        bias_broadcast=bias_broadcast,
        offset_ptr="offset",
    )


@registry.reg("cuda.grouped_fmha_style_b2b_bmm.func_decl")
def grouped_fmha_style_b2b_bmm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.grouped_fmha_style_b2b_bmm.func_call")
def grouped_fmha_style_b2b_bmm_gen_function_call(func_attrs, indent="  "):
    """the function for generating a function call for attention"""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) in (3, 4)

    output_name = func_attrs["outputs"][0]._attrs["name"]
    q_name = func_attrs["inputs"][0]._attrs["name"]
    k_name = func_attrs["inputs"][1]._attrs["name"]
    v_name = func_attrs["inputs"][2]._attrs["name"]

    bias_name = "nullptr"
    if len(func_attrs["inputs"]) == 4:
        bias_name = func_attrs["inputs"][3]._attrs["name"]

    q_shape = func_attrs["inputs"][0]._attrs["shape"]
    jagged_intvar = q_shape[0]
    batch_size_str = jagged_intvar.batch_dim()._attrs["name"]
    if len(jagged_intvar.jagged_dims()) != 1:
        raise RuntimeError(
            "Only support 1 jagged dim in grouped_fmha_style_b2b_bmm for now! "
            f"Current jagged intvar: {jagged_intvar}"
        )
    max_seq_length_dim = jagged_intvar.jagged_dims()[0].max_value()
    max_seq_length_str = (
        str(max_seq_length_dim.value())
        if isinstance(max_seq_length_dim, IntImm)
        else max_seq_length_dim._attrs["name"]
    )
    num_heads_str = q_shape[1]._attrs["name"]
    offset = f"{jagged_intvar.offsets_var_name()}.data[0]"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        query=q_name,
        key=k_name,
        value=v_name,
        bias=bias_name,
        accum_ptr="global_workspace_",
        batch_size=batch_size_str,
        seq_length=max_seq_length_str,
        seq_length_kv=max_seq_length_str,
        num_heads=num_heads_str,
        offset=offset,
        indent=indent,
    )
