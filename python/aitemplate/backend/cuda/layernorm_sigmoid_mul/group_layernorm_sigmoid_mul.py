# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
group LayerNorm_Sigmoid_Mul codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from ... import registry
from ...common import tensor_accessor_codegen
from ...target import Target
from .. import cuda_common
from . import layernorm_common

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "logging.h"

namespace {

{{gamma_beta_const_defs}}

{{tensor_accessor_libs}}
{{custom_libs}}

}  // namespace

{{func_signature}}
{
    {{output_accessor_template}}
    invokeGroupLayernormSigmoidMul<half, float, {{fuse_sigmoid_mul}}, {{num_inputs}}>(output, input, gamma, beta, b, m, n, eps, stream, output_accessors);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(half* output[],
                   half* input[],
                   half* gamma[],
                   half* beta[],
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

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  {{input_elem_type}} *outputs[] = {
{{indent}}    {{outputs}}
{{indent}}  };

{{indent}}  {{input_elem_type}} *inputs[] = {
{{indent}}    {{inputs}}
{{indent}}  };

{{indent}}  {{input_elem_type}} *gamma[] = {
{{indent}}    {{gamma}}
{{indent}}  };

{{indent}}  {{input_elem_type}} *beta[] = {
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
    """[summary]

    Parameters
    ----------

    Returns
    -------
    [type]
        [description]
    """
    output_accessor_decls_str = "\n    ".join(
        tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name="output_accessor_{}".format(i), tensor_accessor=output_accessor
        )
        for i, output_accessor in enumerate(func_attrs["output_accessors"])
    )

    output_accessors = ",\n      ".join(
        [
            "output_accessor_{}".format(i)
            for i in range(len(func_attrs["output_accessors"]))
        ]
    )

    output_accessor_template = OUTPUT_ACCESSORS_TEMPLATE.render(
        output_accessor_decls=output_accessor_decls_str,
        output_accessors=output_accessors,
    )

    op = func_attrs["op"]

    if op == "group_layernorm_sigmoid_mul":
        fuse_sigmoid_mul = "true"
    elif op == "group_layernorm":
        fuse_sigmoid_mul = "false"
    else:
        raise RuntimeError(f"Unsupported op: {op}")

    gamma_beta_const_defs = layernorm_common.gamma_beta_const_defs(func_attrs)
    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "layernorm_sigmoid_mul_kernel.cuh"
        ),
        tensor_accessor_libs=tensor_accessor_codegen.get_libs(),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        fuse_sigmoid_mul=fuse_sigmoid_mul,
        num_inputs=len(func_attrs["outputs"]),
        output_accessor_template=output_accessor_template,
        gamma_beta_const_defs=gamma_beta_const_defs,
    )


@registry.reg("cuda.group_layernorm.func_decl")
@registry.reg("cuda.group_layernorm_sigmoid_mul.func_decl")
def group_layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
    """[summary]

    Parameters
    ----------
    func_name : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.group_layernorm.func_call")
@registry.reg("cuda.group_layernorm_sigmoid_mul.func_call")
def group_layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]
    """

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

    output_ptrs = ",\n        ".join(
        [
            layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(
                name=out._attrs["name"]
            )
            for out in outputs
        ]
    )

    input_ptrs = ",\n        ".join(
        [
            layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(name=i._attrs["name"])
            for i in inputs
        ]
    )

    gamma_strs = (
        ["nullptr"] * b
        if gammas is None
        else (
            [
                layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(
                    name=i._attrs["name"]
                )
                for i in gammas
            ]
        )
    )
    gamma_ptrs = ",\n        ".join(gamma_strs)

    beta_strs = (
        ["nullptr"] * b
        if betas is None
        else (
            [
                layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(
                    name=i._attrs["name"]
                )
                for i in betas
            ]
        )
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
    for i, x in enumerate(inputs):
        shapes = x._attrs["shape"]
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
        input_elem_type=cuda_common.dtype_to_cuda_type(inputs[0]._attrs["dtype"]),
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
