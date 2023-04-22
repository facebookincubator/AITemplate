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
Common codegen functions for group_norm.
"""

import os
from typing import Any, Dict, List

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target

FUNC_SIGNATURE = jinja2.Template(
    """
cudaError_t {{func_name}}(void* output,
                          void* input,
                          void* gamma,
                          void* beta,
                          int N,
                          const float eps,
                          const int max_smem_size,
                          void* workspace,
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
{{indent}}  {{func_name}}(
{{indent}}     {{output}}, {{input}}, {{gamma}}, {{beta}}, {{N}},
{{indent}}     {{eps}}, max_smem_size_, global_workspace_,
{{indent}}  stream /* default stream */
{{indent}}  );
{{indent}}}
    """
)


FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "logging.h"
#include <math_constants.h>
#include <assert.h>

using bfloat16 = __nv_bfloat16;
using bfloat16_2 = __nv_bfloat162;

{{gamma_beta_const_defs}}

namespace {

{{helper_libs}}

{{custom_libs}}

}  // namespace

{{func_signature}}
{

    return invokeGroupNorm<{{elem_input_type}}, {{FuseSwish}}, {{H}}, {{W}}, {{C}}, {{G}}>(
            static_cast<{{elem_input_type}}*>(output),
            static_cast<{{elem_input_type}}*>(input),
            static_cast<{{elem_input_type}}*>(gamma),
            static_cast<{{elem_input_type}}*>(beta),
            N,
            eps,
            max_smem_size,
            workspace,
            stream);
}
    """
)


def get_input_names(func_attrs: Dict[str, Any]) -> List[str]:
    """
    Return a list of rendered name strings for inputs. It returns nullptr
    for gamma and beta if they are None.
    """
    inputs = func_attrs["inputs"]
    x = inputs[0]
    gamma = None
    beta = None

    idx = 1
    if func_attrs["gamma_constant"] is None:
        gamma = inputs[idx]
        idx += 1
    if func_attrs["beta_constant"] is None:
        beta = inputs[idx]
        idx += 1

    input_name = x._attrs["name"]
    if gamma is None:
        gamma_name = "nullptr"
    else:
        gamma_name = gamma._attrs["name"]
    if beta is None:
        beta_name = "nullptr"
    else:
        beta_name = beta._attrs["name"]

    return (input_name, gamma_name, beta_name)


def groupnorm_gen_function(func_attrs: Dict[str, Any]) -> str:
    use_swish = True if "swish" in func_attrs["name"] else False
    input_shape = func_attrs["inputs"][0].shape()

    H = input_shape[1].value()
    W = input_shape[2].value()
    C = input_shape[3].value()
    G = func_attrs["num_groups"]

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        helper_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "layer_norm.cuh"
        ),
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "groupnorm_kernel.cuh"
        ),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
        elem_input_type=elem_input_type,
        FuseSwish="true" if use_swish else "false",
        H=H,
        W=W,
        C=C,
        G=G,
    )


def groupnorm_gen_func_decl(func_attrs: Dict[str, Any]) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


def groupnorm_gen_func_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert 1 <= len(
        func_attrs["inputs"]
    ), "expected at least 1 inputs but got {}".format(len(func_attrs["inputs"]))

    output_name = func_attrs["outputs"][0]._attrs["name"]
    (input_name, gamma_name, beta_name) = get_input_names(func_attrs)
    input_shape = func_attrs["inputs"][0]._attrs["shape"]
    eps = func_attrs["eps"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        N=input_shape[0]._attrs["name"],
        eps=eps,
        indent=indent,
    )
