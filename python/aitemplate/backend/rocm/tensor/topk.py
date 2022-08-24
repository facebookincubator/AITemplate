# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
topk kernel codegen for ROCM.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.tensor import topk_common

# pylint: disable=C0301

header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
"""


@registry.reg("rocm.topk.gen_function")
def topk_gen_function(func_attrs: Dict[str, Any]) -> str:
    return topk_common.gen_function(func_attrs, header_files, ROCMSpec())


@registry.reg("rocm.topk.func_decl")
def topk_gen_function_decl(func_attrs: Dict[str, Any]):
    return topk_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.topk.func_call")
def topk_gen_function_call(func_attrs, indent="  "):
    return topk_common.gen_function_call(func_attrs, ROCMSpec(), indent)


@registry.reg("rocm.topk.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return topk_common.gen_profiler(func_attrs, workdir, header_files, ROCMSpec())
