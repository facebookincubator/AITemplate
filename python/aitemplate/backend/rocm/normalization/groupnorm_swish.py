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
from typing import Any, Dict

from aitemplate.backend import registry

from aitemplate.backend.rocm.normalization.groupnorm import (
    groupnorm_extract_config,
    groupnorm_gen_func_call,
    groupnorm_gen_func_decl,
    groupnorm_gen_function,
    groupnorm_gen_profiler,
)


@registry.reg("rocm.groupnorm_swish.config")
def extract_config(func_attrs):
    return groupnorm_extract_config(func_attrs)


@registry.reg("rocm.groupnorm_swish.gen_profiler")
def gen_profiler(func_attrs: Dict[str, Any], workdir: str, indent: str = "  ") -> str:
    return groupnorm_gen_profiler(func_attrs, workdir, indent, use_swish=True)


@registry.reg("rocm.groupnorm_swish.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    return groupnorm_gen_function(func_attrs, use_swish=True)


@registry.reg("rocm.groupnorm_swish.func_decl")
def func_decl(func_attrs: Dict[str, Any]) -> str:
    return groupnorm_gen_func_decl(func_attrs)


@registry.reg("rocm.groupnorm_swish.func_call")
def gen_func_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return groupnorm_gen_func_call(func_attrs, indent)
