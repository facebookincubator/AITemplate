# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
permute210 for rocm
"""

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.tensor import permute210_common

# pylint: disable=C0301,W0613,W0612

Header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
"""


@registry.reg("rocm.permute210.gen_function")
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
    return permute210_common.gen_function(func_attrs, Header_files, ROCMSpec())


@registry.reg("rocm.permute210.func_decl")
def gen_function_decl(func_attrs):
    return permute210_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.permute210.func_call")
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
    return permute210_common.gen_function_call(func_attrs, ROCMSpec(), indent)
