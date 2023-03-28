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
Graph pass to invoke profiling with dynamic shapes.
"""
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import List

from aitemplate.backend import builder, codegen
from aitemplate.compiler.base import Tensor

# pylint: disable=C0103,W0613,W0102


_LOGGER = logging.getLogger(__name__)


def profile_dynamic_dim(sorted_graph: List[Tensor], workdir="./tmp"):
    _LOGGER.info("Current dynamic profiler supports ONLY ONE dynamic dim.")
    generated_profilers = list(codegen.gen_profiler(sorted_graph, workdir))
    generated_profilers = [p for p in generated_profilers if p is not None]
    compile_engine = builder.Builder()
    compile_engine.make_profilers(generated_profilers, workdir)
    funcs_to_profile = OrderedDict(
        (func._attrs["name"], func)
        for node in sorted_graph
        for func in node.src_ops()
        if func._attrs["has_profiler"]
    )
    for f in funcs_to_profile.values():
        f.profile_dynamic_dim(
            workdir=workdir,
        )
    for node in sorted_graph:
        for func in node.src_ops():
            if func._attrs["has_profiler"]:
                func._attrs["exec_path"] = deepcopy(
                    funcs_to_profile[func._attrs["name"]]._attrs["exec_path"]
                )
