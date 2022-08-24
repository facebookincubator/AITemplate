# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Batch LayerNorm_Sigmoid_Mul codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from ... import registry
from ...common import tensor_accessor_codegen
from ...target import Target
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
    invokeBatchLayernormSigmoidMul<half, float, {{fuse_sigmoid_mul}}>(output, input, gamma, beta, b, m, n, eps, stream);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(half* output,
                   half* input,
                   const half* gamma,
                   const half* beta,
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
    """[summary]

    Parameters
    ----------

    Returns
    -------
    [type]
        [description]
    """
    gamma_beta_const_defs = layernorm_common.gamma_beta_const_defs(func_attrs)
    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "layernorm_sigmoid_mul_kernel.cuh"
        ),
        tensor_accessor_libs=tensor_accessor_codegen.get_libs(),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        fuse_sigmoid_mul="true",
        gamma_beta_const_defs=gamma_beta_const_defs,
    )


@registry.reg("cuda.batch_layernorm_sigmoid_mul.func_decl")
def batch_layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
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


@registry.reg("cuda.batch_layernorm_sigmoid_mul.func_call")
def batch_layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
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
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert (
        1 <= len(func_attrs["inputs"]) <= 4
    ), "expected 1 ~ 4 inputs but got {}".format(len(func_attrs["inputs"]))

    output_name = layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )
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
