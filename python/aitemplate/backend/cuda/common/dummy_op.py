# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Dummy op codegen for CUDA.
"""

from typing import Any, Dict

from ... import registry


@registry.reg("cuda.size.gen_function")
def dummy_gen_function(func_attrs: Dict[str, Any]) -> str:
    return ""


@registry.reg("cuda.size.func_decl")
def dummy_gen_function_decl(func_attrs):
    return ""


@registry.reg("cuda.size.func_call")
def dummy_gen_function_call(func_attrs, indent):
    return ""
