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
Dummy op codegen for CUDA.
"""

from typing import Any, Dict

from aitemplate.backend import registry


@registry.reg("cuda.size.gen_function")
def dummy_gen_function(func_attrs: Dict[str, Any]) -> str:
    return ""


@registry.reg("cuda.size.func_decl")
def dummy_gen_function_decl(func_attrs):
    return ""


@registry.reg("cuda.size.func_call")
def dummy_gen_function_call(func_attrs, indent):
    return ""
