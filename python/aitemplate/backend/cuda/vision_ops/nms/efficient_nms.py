# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
efficient_nms kernel codegen for CUDA.
"""

from typing import Any, Dict

from .... import registry
from ....backend_spec import CUDASpec
from ....common.vision_ops import efficient_nms_common

# pylint: disable=C0301

func_header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cuda_runtime_api.h"
#include "cub/cub.cuh"
"""

profiler_header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
"""


@registry.reg("cuda.efficient_nms.gen_function")
def efficient_nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    return efficient_nms_common.gen_function(func_attrs, func_header_files, CUDASpec())


@registry.reg("cuda.efficient_nms.func_decl")
def efficient_nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return efficient_nms_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.efficient_nms.func_call")
def efficient_nms_gen_function_call(func_attrs, indent="  "):
    return efficient_nms_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.efficient_nms.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return efficient_nms_common.gen_profiler(
        func_attrs, workdir, profiler_header_files, CUDASpec()
    )
