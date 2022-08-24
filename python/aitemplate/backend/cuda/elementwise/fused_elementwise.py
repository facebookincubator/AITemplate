# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Elementwise codegen for CUDA.
"""

import os
from typing import Any, Dict

from ... import registry
from ...backend_spec import CUDASpec
from ...common import elementwise_common
from ...target import Target

HEAD_TEMPLATE = """
#include <cuda_fp16.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/constants.h"
"""


@registry.reg("cuda.fused_elementwise.gen_function")
def fused_elementwise_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates fused_elementwise function definition."""
    custom_libs = Target.current().get_custom_libs(
        os.path.dirname(__file__), "custom_math.cuh"
    )
    return elementwise_common.fused_elementwise_gen_function(
        func_attrs=func_attrs,
        custom_libs=custom_libs,
        head_template=HEAD_TEMPLATE,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.fused_elementwise.func_decl")
def fused_elementwise_gen_function_decl(func_attrs):
    """Generates fused_elementwise function declaration."""
    return elementwise_common.fused_elementwise_gen_function_decl(
        func_attrs=func_attrs,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.fused_elementwise.func_call")
def fused_elementwise_gen_function_call(func_attrs, indent):
    """Generates fused_elementwise function call."""
    return elementwise_common.fused_elementwise_gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        backend_spec=CUDASpec(),
    )
