# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
argmax kernel codegen for ROCM.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.tensor import argmax_common

# pylint: disable=C0301

header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
"""


@registry.reg("rocm.argmax.gen_function")
def argmax_gen_function(func_attrs: Dict[str, Any]) -> str:
    return argmax_common.gen_function(func_attrs, header_files, ROCMSpec())


@registry.reg("rocm.argmax.func_decl")
def argmax_gen_function_decl(func_attrs: Dict[str, Any]):
    return argmax_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.argmax.func_call")
def argmax_gen_function_call(func_attrs, indent="  "):
    return argmax_common.gen_function_call(func_attrs, ROCMSpec(), indent)


@registry.reg("rocm.argmax.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return argmax_common.gen_profiler(func_attrs, workdir, header_files, ROCMSpec())
