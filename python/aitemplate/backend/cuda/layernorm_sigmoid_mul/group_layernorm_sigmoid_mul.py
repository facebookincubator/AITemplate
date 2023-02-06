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
group LayerNorm_Sigmoid_Mul codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from ... import registry
from ...backend_spec import CUDASpec
from ...common import tensor_accessor_codegen
from ...target import Target
from . import layernorm_common

# pylint: disable=C0301


LOCAL_PARAM_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}} {{elem_input_type}} *{{local_param_name}}[{{num_groups}}] = {
{% for i in range(num_groups) %}
{{indent}}    static_cast<{{elem_input_type}}*>({{param_name}}[{{i}}]){{", " if not loop.last else ""}}
{% endfor %}
{{indent}}
{{indent}}};
"""
)


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
    {{output_accessor_template}}
    {{input_accessor_template}}
    {{local_param_defs}}
    invokeGroupLayernormSigmoidMul<{{elem_input_type}}, float, {{fuse_sigmoid_mul}}, {{num_inputs}}>(
        {{local_param_names}},
        b, m, n, eps, stream, input_accessors, output_accessors);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output[],
                   void* input[],
                   void* gamma[],
                   void* beta[],
                   int b,
                   int m,
                   int64_t* n,
                   float eps,
                   cudaStream_t stream)
    """
)

OUTPUT_ACCESSORS_TEMPLATE = jinja2.Template(
    """
    {{output_accessor_decls}}
    TensorAccessor output_accessors[] = {
        {{output_accessors}}
    };
    """
)

INPUT_ACCESSORS_TEMPLATE = jinja2.Template(
    """
    {{input_accessor_decls}}
    TensorAccessor input_accessors[] = {
        {{input_accessors}}
    };
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

{{indent}}  void *outputs[] = {
{{indent}}    {{outputs}}
{{indent}}  };

{{indent}}  void *inputs[] = {
{{indent}}    {{inputs}}
{{indent}}  };

{{indent}}  void *gamma[] = {
{{indent}}    {{gamma}}
{{indent}}  };

{{indent}}  void *beta[] = {
{{indent}}    {{beta}}
{{indent}}  };

{{indent}}  {{m_n_shape_func}}
{{indent}}  int64_t n[{{num_inputs}}] = {
{{indent}}    {{n}}
{{indent}}  };


{{indent}}  {{func_name}}(
{{indent}}     outputs, inputs, gamma, beta,
{{indent}}     {{b}}, {{m}}, n, {{eps}}, stream /* default stream */
{{indent}}  );
{{indent}}}
    """
)


@registry.reg("cuda.group_layernorm.gen_function")
@registry.reg("cuda.group_layernorm_sigmoid_mul.gen_function")
def group_layernorm_sigmoid_mul_gen_function(func_attrs: Dict[str, Any]) -> str:
    output_accessor_decls_str = "\n    ".join(
        tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name=f"output_accessor_{i}", tensor_accessor=output_accessor
        )
        for i, output_accessor in enumerate(func_attrs["output_accessors"])
    )

    input_accessor_decls_str = "\n    ".join(
        tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name=f"input_accessor_{i}", tensor_accessor=input_accessor
        )
        for i, input_accessor in enumerate(func_attrs["input_accessors"])
    )

    output_accessors = ",\n      ".join(
        [f"output_accessor_{i}" for i in range(len(func_attrs["output_accessors"]))]
    )

    input_accessors = ",\n      ".join(
        [f"input_accessor_{i}" for i in range(len(func_attrs["input_accessors"]))]
    )

    output_accessor_template = OUTPUT_ACCESSORS_TEMPLATE.render(
        output_accessor_decls=output_accessor_decls_str,
        output_accessors=output_accessors,
    )

    input_accessor_template = INPUT_ACCESSORS_TEMPLATE.render(
        input_accessor_decls=input_accessor_decls_str,
        input_accessors=input_accessors,
    )

    op = func_attrs["op"]

    if op == "group_layernorm_sigmoid_mul":
        fuse_sigmoid_mul = "true"
    elif op == "group_layernorm":
        fuse_sigmoid_mul = "false"
    else:
        raise RuntimeError(f"Unsupported op: {op}")

    gamma_beta_const_defs = layernorm_common.gamma_beta_const_defs(func_attrs)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    num_groups = len(func_attrs["outputs"])
    params = ["output", "input", "gamma", "beta"]
    local_param_defs = []
    local_param_names = []
    for param in params:
        local_name = f"{param}_tmp"
        local_param_def = LOCAL_PARAM_DEF_TEMPLATE.render(
            indent="  ",
            elem_input_type=elem_input_type,
            num_groups=num_groups,
            local_param_name=local_name,
            param_name=param,
        )
        local_param_defs.append(local_param_def)
        local_param_names.append(local_name)
    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "layernorm_sigmoid_mul_kernel.cuh"
        ),
        tensor_accessor_libs=tensor_accessor_codegen.get_libs(),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        fuse_sigmoid_mul=fuse_sigmoid_mul,
        num_inputs=len(func_attrs["outputs"]),
        output_accessor_template=output_accessor_template,
        input_accessor_template=input_accessor_template,
        gamma_beta_const_defs=gamma_beta_const_defs,
        local_param_defs="\n".join(local_param_defs),
        local_param_names=",".join(local_param_names),
    )


@registry.reg("cuda.group_layernorm.func_decl")
@registry.reg("cuda.group_layernorm_sigmoid_mul.func_decl")
def group_layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.group_layernorm.func_call")
@registry.reg("cuda.group_layernorm_sigmoid_mul.func_call")
def group_layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
    # [inputs, gamma, beta]
    all_inputs = func_attrs["inputs"]
    b = len(func_attrs["outputs"])
    gammas = None
    betas = None
    inputs = all_inputs[:b]
    idx = b
    if func_attrs["gamma_constant"] is None:
        gammas = all_inputs[idx : idx + b]
        idx += b
    if func_attrs["beta_constant"] is None:
        betas = all_inputs[idx : idx + b]
        idx += b
    outputs = func_attrs["outputs"]

    output_ptrs = ",\n        ".join([out._attrs["name"] for out in outputs])

    input_ptrs = ",\n        ".join([i._attrs["name"] for i in inputs])

    gamma_strs = (
        ["nullptr"] * b if gammas is None else ([i._attrs["name"] for i in gammas])
    )
    gamma_ptrs = ",\n        ".join(gamma_strs)

    beta_strs = (
        ["nullptr"] * b if betas is None else ([i._attrs["name"] for i in betas])
    )
    beta_ptrs = ",\n        ".join(beta_strs)

    all_shape_funcs = []
    # all Ms are the same
    input_0_shapes = inputs[0]._attrs["shape"]
    norm_ndim = len(func_attrs["normalized_shape"][0])
    m_name = "M"
    m_shape_func = layernorm_common.generate_m_shape_func(
        input_0_shapes,
        norm_ndim,
        m_name,
        indent + "    ",
    )
    all_shape_funcs.append(m_shape_func)

    n = []
    input_accessors = func_attrs["input_accessors"]
    for i, acc in enumerate(input_accessors):
        shapes = acc.original_shapes
        n_name = f"N{i}"
        n_shape_func = layernorm_common.generate_n_shape_func(
            shapes,
            norm_ndim,
            n_name,
            indent + "    ",
        )
        all_shape_funcs.append(n_shape_func)
        n.append(n_name)

    n_str = ", ".join([str(i) for i in n])

    eps = func_attrs["eps"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        m_n_shape_func="".join(all_shape_funcs),
        indent=indent,
        outputs=output_ptrs,
        inputs=input_ptrs,
        gamma=gamma_ptrs,
        beta=beta_ptrs,
        n=n_str,
        b=b,
        m=m_name,
        eps=eps,
    )
