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
Batch LayerNorm_Sigmoid_Mul codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import tensor_accessor_codegen
from aitemplate.backend.cuda.layernorm_sigmoid_mul import layernorm_common
from aitemplate.backend.target import Target

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "logging.h"

using bfloat16 = __nv_bfloat16;

namespace {

{{gamma_beta_const_defs}}

{{tensor_accessor_libs}}
{{custom_libs}}

}  // namespace

{{func_signature}}
{
    invokeBatchLayernormSigmoidMul<{{elem_input_type}}, float, {{fuse_sigmoid_mul}}>(
        static_cast<{{elem_input_type}}*>(output),
        static_cast<{{elem_input_type}}*>(input),
        static_cast<const {{elem_input_type}}*>(gamma),
        static_cast<const {{elem_input_type}}*>(beta),
        b, m, n, eps, stream);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   void* input,
                   const void* gamma,
                   const void* beta,
                   int b,
                   int m,
                   int n,
                   float eps,
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
{{indent}}   {{output}}, {{input}}, {{gamma}}, {{beta}},
{{indent}}   {{b}}, {{m}}, {{n}}, {{eps}}, stream /* default stream */
{{indent}});
    """
)


@registry.reg("cuda.batch_layernorm_sigmoid_mul.gen_function")
def batch_layernorm_sigmoid_mul_gen_function(func_attrs: Dict[str, Any]) -> str:
    gamma_beta_const_defs = layernorm_common.gamma_beta_const_defs(func_attrs)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "layernorm_sigmoid_mul_kernel.cuh"
        ),
        tensor_accessor_libs=tensor_accessor_codegen.get_libs(),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        fuse_sigmoid_mul="true",
        gamma_beta_const_defs=gamma_beta_const_defs,
    )


@registry.reg("cuda.batch_layernorm_sigmoid_mul.func_decl")
def batch_layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.batch_layernorm_sigmoid_mul.func_call")
def batch_layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert (
        1 <= len(func_attrs["inputs"]) <= 4
    ), "expected 1 ~ 4 inputs but got {}".format(len(func_attrs["inputs"]))

    output_name = func_attrs["outputs"][0]._attrs["name"]
    (input_name, gamma_name, beta_name) = layernorm_common.get_input_names(func_attrs)

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    assert len(shapes) == 3

    (b_name, m_name, n_name) = (shape._attrs["name"] for shape in shapes)

    eps = func_attrs["eps"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        b=b_name,
        m=m_name,
        n=n_name,
        eps=eps,
        indent=indent,
    )
