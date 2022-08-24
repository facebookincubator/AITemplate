# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from copy import deepcopy
from typing import List

from ...backend import codegen
from ...utils import logger
from ..base import Tensor

# pylint: disable=C0103,W0613,W0102


def profile_dynamic_dim(sorted_graph: List[Tensor], workdir="./tmp"):
    """[summary]

    Parameters
    ----------
    sorted_graph : List[Tensor]
        [description]
    workdir : str, optional
        [description], by default "./tmp"
    """
    logger.info(__name__, "Current dynamic profiler supports ONLY ONE dynamic dim.")
    codegen.gen_profiler(sorted_graph, workdir)
    profiled = {}
    for node in sorted_graph:
        for func in node.src_ops():
            func_name = func._attrs["name"]
            if func_name in profiled:
                # paths = profiled[func_name]._attrs["exec_path"].keys()
                func._attrs["exec_path"] = deepcopy(
                    profiled[func_name]._attrs["exec_path"]
                )
                # for path in paths:
                #     func._attrs["exec_path"][path] = \
                #         profiled[func_name]._attrs["exec_path"][path]
                continue
            if func._attrs["has_profiler"]:
                func.profile_dynamic_dim(workdir=workdir)
                profiled[func_name] = func
