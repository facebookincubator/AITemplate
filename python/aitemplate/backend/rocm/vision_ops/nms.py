# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
nms kernel codegen for ROCM.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.vision_ops import nms_common

# pylint: disable=C0301

header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
"""


@registry.reg("rocm.nms.gen_function")
def nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    return nms_common.gen_function(func_attrs, header_files, ROCMSpec())


@registry.reg("rocm.nms.func_decl")
def nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return nms_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.nms.func_call")
def nms_gen_function_call(func_attrs, indent="  "):
    return nms_common.gen_function_call(func_attrs, ROCMSpec(), indent)


@registry.reg("rocm.nms.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return nms_common.gen_profiler(func_attrs, workdir, header_files, ROCMSpec())
