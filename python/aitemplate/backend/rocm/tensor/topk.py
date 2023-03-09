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
topk kernel codegen for ROCM.
"""

from typing import Any, Dict

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec
from aitemplate.backend.common.tensor import topk_common

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
