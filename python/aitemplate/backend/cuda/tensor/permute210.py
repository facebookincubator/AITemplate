# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
permute210 for cuda
"""

from ... import registry
from ...backend_spec import CUDASpec
from ...common.tensor import permute210_common

# pylint: disable=C0301,W0613,W0612

Header_files = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
"""


@registry.reg("cuda.permute210.gen_function")
def gen_function(func_attrs, template_path):
    """

    Parameters
    ----------
    func_attrs : dict
        Attributes from Operator
    template_path : str
        path to library used

    Returns
    -------
    str
        Source code for function generated.
    """
    return permute210_common.gen_function(func_attrs, Header_files, CUDASpec())


@registry.reg("cuda.permute210.func_decl")
def gen_function_decl(func_attrs):
    return permute210_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.permute210.func_call")
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
    return permute210_common.gen_function_call(func_attrs, CUDASpec(), indent)
