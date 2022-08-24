# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
batch_gather kernel codegen for ROCM.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import ROCMSpec
from ...common.tensor import batch_gather_common

# pylint: disable=C0301

header_files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
"""


@registry.reg("rocm.batch_gather.gen_function")
def batch_gather_gen_function(func_attrs: Dict[str, Any]) -> str:
    return batch_gather_common.gen_function(func_attrs, header_files, ROCMSpec())


@registry.reg("rocm.batch_gather.func_decl")
def batch_gather_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return batch_gather_common.gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.batch_gather.func_call")
def batch_gather_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return batch_gather_common.gen_function_call(func_attrs, indent)
