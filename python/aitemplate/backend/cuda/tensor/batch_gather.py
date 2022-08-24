# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
batch_gather kernel codegen for CUDA.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import CUDASpec
from ...common.tensor import batch_gather_common

# pylint: disable=C0301

header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
"""


@registry.reg("cuda.batch_gather.gen_function")
def batch_gather_gen_function(func_attrs: Dict[str, Any]) -> str:
    return batch_gather_common.gen_function(func_attrs, header_files, CUDASpec())


@registry.reg("cuda.batch_gather.func_decl")
def batch_gather_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return batch_gather_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.batch_gather.func_call")
def batch_gather_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return batch_gather_common.gen_function_call(func_attrs, indent, True)
