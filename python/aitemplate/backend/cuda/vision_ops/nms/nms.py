# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
nms kernel codegen for CUDA.
"""

from typing import Any, Dict

from .... import registry
from ....backend_spec import CUDASpec
from ....common.vision_ops import nms_common

# pylint: disable=C0301

header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
"""


@registry.reg("cuda.nms.gen_function")
def nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    return nms_common.gen_function(func_attrs, header_files, CUDASpec())


@registry.reg("cuda.nms.func_decl")
def nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return nms_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.nms.func_call")
def nms_gen_function_call(func_attrs, indent="  "):
    return nms_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.nms.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return nms_common.gen_profiler(func_attrs, workdir, header_files, CUDASpec())
