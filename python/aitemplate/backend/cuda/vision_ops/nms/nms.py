#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
nms kernel codegen for CUDA.
"""

from typing import Any, Dict

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common.vision_ops import nms_common

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
