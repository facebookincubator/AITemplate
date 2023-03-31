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
Graph pass to invoke profiling.
"""
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import List

from aitemplate.backend import builder, codegen

from aitemplate.backend.profiler_runner import ProfilerRunner
from aitemplate.backend.target import Target
from aitemplate.compiler.base import DynamicProfileStrategy, Tensor

from aitemplate.compiler.ops.gemm_universal.gemm_common import (
    gemm,
    GemmProfilerPostprocessingDelegate,
)

# pylint: disable=C0103,W0613,W0102


_LOGGER = logging.getLogger(__name__)


def elapsed_dt_sec(start_t_sec):
    return datetime.now() - start_t_sec


def _splitter(data, pred=bool):
    group_a = []
    group_b = []
    for d in data:
        (group_a if pred(d) else group_b).append(d)
    return group_a, group_b


def profile(
    sorted_graph: List[Tensor],
    workdir="./tmp",
    devices=None,
    dynamic_profiling_strategy=DynamicProfileStrategy.MAX,
):
    """Profiles kernels.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        A sorted graph which contains all functions for profiling.
    workdir : str, optional
        The base dir to generate profiling source codes. By default "./tmp"
    devices: list, optional
        A list of device ids which can be used for profiling.
        By default device 0 will be used.
    dynamic_profiling_strategy: DynamicProfileStrategy, optional
        A dynamic profiling strategy, used to filter generated profiles at compile time.
        See also: :func:`~aitemplate.compiler.transform.profile.profile`
        By default MAX is used, i.e. to profile a dynamic range, an upper bound will be used.
    """

    if devices is None:
        devices = [0]
    profiler_dir = os.path.join(workdir)
    start_t = datetime.now()
    generated_profilers = list(
        codegen.gen_profiler(sorted_graph, profiler_dir, dynamic_profiling_strategy)
    )
    generated_profilers = [p for p in generated_profilers if p is not None]
    _LOGGER.info(
        f"generated {len(generated_profilers)} profilers elapsed time: {elapsed_dt_sec(start_t)}",
    )
    start_t = datetime.now()
    compile_engine = builder.Builder()
    compile_engine.make_profilers(generated_profilers, profiler_dir)
    _LOGGER.info(f"compiled profilers elapsed time: {elapsed_dt_sec(start_t)}")
    funcs_to_profile = OrderedDict(
        (func._attrs["name"], func)
        for node in sorted_graph
        for func in node.src_ops()
        if func._attrs["has_profiler"]
    )

    start_t = datetime.now()
    gemms, non_gemms = _splitter(
        funcs_to_profile.values(), lambda f: isinstance(f, gemm)
    )
    for f in non_gemms:
        f.profile(
            workdir=profiler_dir,
            devices=devices,
        )
    timeout = 2400 if Target.current().name() == "rocm" else 240
    profiler_runner = ProfilerRunner(
        devices,
        timeout=timeout,
        postprocessing_delegate=GemmProfilerPostprocessingDelegate(),
    )
    for f in gemms:
        f.profile(
            workdir=profiler_dir,
            profiler_runner=profiler_runner,
        )
    profiler_runner.join()
    _LOGGER.info(
        f"ran {len(funcs_to_profile)} profilers elapsed time: {elapsed_dt_sec(start_t)}",
    )
    for node in sorted_graph:
        for func in node.src_ops():
            if func._attrs["has_profiler"]:
                func._attrs["exec_path"] = deepcopy(
                    funcs_to_profile[func._attrs["name"]]._attrs["exec_path"]
                )
