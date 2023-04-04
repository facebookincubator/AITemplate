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
bert_embeddings kernel codegen for CUDA.
"""

import math
from typing import Any, Dict

import jinja2

from ... import registry
from ...backend_spec import ROCMSpec

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include "logging.h"
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_sparse_embeddings_forward_layernorm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#define EMBEDDING_DIM {{embedding_dim}}

using EmbElementwiseOperation = ck::tensor_operation::element_wise::AddAdd;
using EmbType = {{elem_input_type}};
using IndexType = {{index_type}};

{{func_signature}}
{
  auto device_instance = ck::tensor_operation::device::DeviceSparseEmbeddingsForwardLayernorm<EmbType, IndexType, EmbType, EmbType, float, EmbType, EmbElementwiseOperation, 256, 1, 256, 1, EMBEDDING_DIM, 1, {{row_v_size}}, 3>{};
  auto argument_ptr = device_instance.MakeArgumentPointer(output,
                                                          {ck::type_convert<EmbType*>(word_embeddings),
                                                          ck::type_convert<EmbType*>(token_type_embeddings),
                                                          ck::type_convert<EmbType*>(position_embeddings)},
                                                          {ck::type_convert<IndexType*>(input_ids),
                                                          ck::type_convert<IndexType*>(token_type_ids),
                                                          ck::type_convert<IndexType*>(position_ids)},
                                                          gamma,
                                                          beta,
                                                          EMBEDDING_DIM,
                                                          indices_num,
                                                          eps,
                                                          EmbElementwiseOperation{});
  if(!device_instance.IsSupportedArgument(argument_ptr.get())){
    LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Embedding problem.";
  }
  auto invoker_ptr = device_instance.MakeInvokerPointer();
  invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
  return;
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   {{index_type}}* input_ids,
                   {{index_type}}* token_type_ids,
                   {{index_type}}* position_ids,
                   void* word_embeddings,
                   void* token_type_embeddings,
                   void* position_embeddings,
                   void* gamma,
                   void* beta,
                   const int64_t indices_num,
                   const float eps,
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
{{indent}}{
{{indent}}  {{calculate_indices_num}}
{{indent}}  {{func_name}}(
{{indent}}            {{output}},
{{indent}}            {{input_ids}},
{{indent}}            {{token_type_ids}},
{{indent}}            {{position_ids}},
{{indent}}            {{word_embeddings}},
{{indent}}            {{token_type_embeddings}},
{{indent}}            {{position_embeddings}},
{{indent}}            {{gamma}},
{{indent}}            {{beta}},
{{indent}}            {{indices_num}},
{{indent}}            {{eps}},
{{indent}}            stream /* default stream */
{{indent}} );

{{indent}}}
    """
)

INDICES_NUM_TEMPLATE = jinja2.Template(
    """
  int64_t indices_num = 1;
  {% for dim_name in dim_names %}
  indices_num *= {{dim_name}};
  {% endfor %}
  """
)


def python_int_dtype_to_c_dtype(dtype):
    if dtype == "int64":
        return "int64_t"
    if dtype in ["int", "int32"]:
        return "int32_t"
    return dtype


@registry.reg("rocm.bert_embeddings.gen_function")
def bert_embeddings_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = ROCMSpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][3]._attrs["dtype"]
    )
    (
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
    ) = func_attrs["inputs"]
    embedding_dim = word_embeddings._size(-1).value()
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_TEMPLATE.render(
        index_type=dtype,
        elem_input_type=elem_input_type,
        embedding_dim=embedding_dim,
        row_v_size=math.gcd(8, embedding_dim // 256),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip(),
    )


@registry.reg("rocm.bert_embeddings.func_decl")
def bert_embeddings_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    dtype = python_int_dtype_to_c_dtype(func_attrs["inputs"][0]._attrs["dtype"])
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            index_type=dtype,
        ).strip()
    )


FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")
FUNC_CALL_INT32_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int32_t*>({{name}})")


def get_int_param_template(tensor):
    name = tensor._attrs["name"]
    dtype = tensor._attrs["dtype"]
    if dtype == "int64":
        return FUNC_CALL_INT64_PARAM_TEMPLATE.render(name=name)
    elif dtype in ("int", "int32"):
        return FUNC_CALL_INT32_PARAM_TEMPLATE.render(name=name)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")


@registry.reg("rocm.bert_embeddings.func_call")
def bert_embeddings_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    (
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        gamma,
        beta,
    ) = func_attrs["inputs"]

    indices_dims = [shape._attrs["name"] for shape in input_ids.shape()]
    indices_num_str = INDICES_NUM_TEMPLATE.render(
        dim_names=indices_dims,
    )

    eps = func_attrs["eps"]
    output_str = func_attrs["outputs"][0]._attrs["name"]

    input_ids_str = get_int_param_template(input_ids)
    token_type_ids_str = get_int_param_template(token_type_ids)
    position_ids_str = get_int_param_template(position_ids)

    word_embeddings_str = word_embeddings._attrs["name"]
    token_type_embeddings_str = token_type_embeddings._attrs["name"]
    position_embeddings_str = position_embeddings._attrs["name"]

    gamma_str = gamma._attrs["name"]
    beta_str = beta._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        calculate_indices_num=indices_num_str,
        output=output_str,
        input_ids=input_ids_str,
        token_type_ids=token_type_ids_str,
        position_ids=position_ids_str,
        word_embeddings=word_embeddings_str,
        token_type_embeddings=token_type_embeddings_str,
        position_embeddings=position_embeddings_str,
        gamma=gamma_str,
        beta=beta_str,
        indices_num="indices_num",
        eps=eps,
        indent=indent,
    )
