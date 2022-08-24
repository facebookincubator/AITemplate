# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
argmax kernel codegen for CUDA.
"""

from typing import Any, Dict

from ... import registry
from ...backend_spec import CUDASpec
from ...common.tensor import argmax_common

# pylint: disable=C0301

header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
"""


@registry.reg("cuda.argmax.gen_function")
def argmax_gen_function(func_attrs: Dict[str, Any]) -> str:
    return argmax_common.gen_function(func_attrs, header_files, CUDASpec())


@registry.reg("cuda.argmax.func_decl")
def argmax_gen_function_decl(func_attrs: Dict[str, Any]):
    return argmax_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.argmax.func_call")
def argmax_gen_function_call(func_attrs, indent="  "):
    return argmax_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.argmax.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return argmax_common.gen_profiler(func_attrs, workdir, header_files, CUDASpec())
