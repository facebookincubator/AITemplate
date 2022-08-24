# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import os
from typing import List

from ...backend import codegen
from ..base import DynamicProfileStrategy, Tensor

# pylint: disable=C0103,W0613,W0102


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
    codegen.gen_profiler(sorted_graph, profiler_dir, dynamic_profiling_strategy)
    profiled = {}
    for node in sorted_graph:
        for func in node.src_ops():
            func_name = func._attrs["name"]
            if func_name in profiled:
                paths = func._attrs["exec_path"].keys()
                for path in paths:
                    func._attrs["exec_path"][path] = profiled[func_name]._attrs[
                        "exec_path"
                    ][path]
                continue
            if func._attrs["has_profiler"]:
                func.profile(
                    workdir=profiler_dir,
                    devices=devices,
                    dynamic_profiling_strategy=dynamic_profiling_strategy,
                )
                profiled[func_name] = func
