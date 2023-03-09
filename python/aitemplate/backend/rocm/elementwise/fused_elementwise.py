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
Elementwise codegen for ROCM.
"""

import os
from typing import Any, Dict

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec
from aitemplate.backend.common import elementwise_common
from aitemplate.backend.target import Target


HEAD_TEMPLATE = """
#include <hip/math_functions.h>
#include <hip/device_functions.h>
"""


@registry.reg("rocm.fused_elementwise.gen_function")
def fused_elementwise_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates fused_elementwise function definition."""

    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "custom_math.h"
    )
    return elementwise_common.fused_elementwise_gen_function(
        func_attrs=func_attrs,
        custom_libs=custom_libs,
        head_template=HEAD_TEMPLATE,
        backend_spec=ROCMSpec(),
    )


@registry.reg("rocm.fused_elementwise.func_decl")
def fused_elementwise_gen_function_decl(func_attrs):
    """Generates fused_elementwise function declaration."""
    return elementwise_common.fused_elementwise_gen_function_decl(
        func_attrs=func_attrs,
        backend_spec=ROCMSpec(),
    )


@registry.reg("rocm.fused_elementwise.func_call")
def fused_elementwise_gen_function_call(func_attrs, indent):
    """Generates fused_elementwise function call."""
    return elementwise_common.fused_elementwise_gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        backend_spec=ROCMSpec(),
    )
