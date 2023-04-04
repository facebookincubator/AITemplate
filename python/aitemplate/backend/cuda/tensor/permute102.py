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
permute102 for cuda
"""

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.tensor import permute102_common

# pylint: disable=C0301,W0613,W0612

Header_files = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"

using bfloat16 = __nv_bfloat16;
"""


@registry.reg("cuda.permute102.gen_function")
def gen_function(func_attrs, template_path):
    """
    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Attributes from Operator
    template_path : str
        path to library used

    Returns
    -------
    str
        Source code for function generated.
    """
    return permute102_common.gen_function(
        func_attrs,
        template_path,
        Header_files,
        CUDASpec(),
    )


@registry.reg("cuda.permute102.func_decl")
def gen_function_decl(func_attrs):
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator

    Returns
    -------
    str
        Function declaration
    """
    return permute102_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.permute102.func_call")
def gen_function_call(func_attrs, indent="  "):
    """
    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    indent : str, optional
        Indentation for function call template, by default "  "

    Returns
    -------
    str
        Driver code for invoking call
    """
    return permute102_common.gen_function_call(func_attrs, CUDASpec(), indent)
