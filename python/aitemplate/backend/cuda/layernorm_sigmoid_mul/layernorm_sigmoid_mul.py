# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
LayerNorm_Sigmoid_Mul codegen for CUDA.
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

{{gamma_beta_const_defs}}

namespace {

{{tensor_accessor_libs}}
{{custom_libs}}

}  // namespace

{{func_signature}}
{
    {{output_accessor}}
    invokeLayernormSigmoidMul<half, float, {{fuse_sigmoid_mul}}>(output, input, gamma, beta, m, n, eps, stream, output_accessor);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(half* output,
                   half* input,
                   const half* gamma,
                   const half* beta,
                   int m,
                   int n,
                   const float eps,
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
{{indent}}{
{{indent}}  {{m_n_shape_func}}
{{indent}}  {{func_name}}(
{{indent}}     {{output}}, {{input}}, {{gamma}}, {{beta}},
{{indent}}     {{m}}, {{n}}, {{eps}}, stream /* default stream */
{{indent}}  );
{{indent}}}
    """
)


@registry.reg("cuda.layernorm.gen_function")
def layernorm_gen_function(func_attrs: Dict[str, Any]) -> str:
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
        fuse_sigmoid_mul="false",
        output_accessor=tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name="output_accessor", tensor_accessor=func_attrs["output_accessors"][0]
        ),
        gamma_beta_const_defs=gamma_beta_const_defs,
    )


@registry.reg("cuda.layernorm_sigmoid_mul.gen_function")
def layernorm_sigmoid_mul_gen_function(func_attrs: Dict[str, Any]) -> str:
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
        output_accessor=tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
            name="output_accessor", tensor_accessor=func_attrs["output_accessors"][0]
        ),
        gamma_beta_const_defs=gamma_beta_const_defs,
    )


@registry.reg("cuda.layernorm.func_decl")
@registry.reg("cuda.layernorm_sigmoid_mul.func_decl")
def layernorm_sigmoid_mul_gen_function_decl(func_attrs: Dict[str, Any]):
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


@registry.reg("cuda.layernorm.func_call")
@registry.reg("cuda.layernorm_sigmoid_mul.func_call")
def layernorm_sigmoid_mul_gen_function_call(func_attrs, indent="  "):
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
    assert 1 <= len(
        func_attrs["inputs"]
    ), "expected at least 1 inputs but got {}".format(len(func_attrs["inputs"]))

    output_name = layernorm_common.FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )
    (input_name, gamma_name, beta_name) = layernorm_common.get_input_names(func_attrs)

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    norm_ndim = len(func_attrs["normalized_shape"])
    m_name = "M"

    m_shape_func = layernorm_common.generate_m_shape_func(
        shapes,
        norm_ndim,
        m_name,
        indent + "    ",
    )

    n_name = "N"
    n_shape_func = layernorm_common.generate_n_shape_func(
        shapes,
        norm_ndim,
        n_name,
        indent + "    ",
    )

    eps = func_attrs["eps"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        m_n_shape_func=m_shape_func + n_shape_func,
        output=output_name,
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        m=m_name,
        n=n_name,
        eps=eps,
        indent=indent + "  ",
    )
