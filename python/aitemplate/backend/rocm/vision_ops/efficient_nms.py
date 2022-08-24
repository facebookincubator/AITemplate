# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
efficient_nms kernel codegen for ROCM.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.vision_ops import efficient_nms_common

# pylint: disable=C0301

header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
"""


@registry.reg("rocm.efficient_nms.gen_function")
def efficient_nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    return efficient_nms_common.gen_function(func_attrs, header_files, ROCMSpec())


@registry.reg("rocm.efficient_nms.func_decl")
def efficient_nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return efficient_nms_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.efficient_nms.func_call")
def efficient_nms_gen_function_call(func_attrs, indent="  "):
    return efficient_nms_common.gen_function_call(func_attrs, ROCMSpec(), indent)


@registry.reg("rocm.efficient_nms.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return efficient_nms_common.gen_profiler(
        func_attrs, workdir, header_files, ROCMSpec()
    )
