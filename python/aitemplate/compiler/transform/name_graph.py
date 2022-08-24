# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import re
from typing import List

from ..base import Tensor

# pylint: disable=C0103

# Make these variables global to allow repeately calling name_graph().
func_cnt = 0
tensor_cnt = 0
func_name_to_tensor_cnt = {}

MEMO = set()


def valid_c_name(name):
    return re.sub(r"\W|^(?=\d)", "_", name)


def unique_name(name):
    name = valid_c_name(name)
    global MEMO
    if name in MEMO:
        return f"{name}_{str(len(MEMO))}"
    else:
        MEMO.add(name)
        return name


def name_graph(sorted_graph: List[Tensor]) -> None:
    """[summary]

    Parameters
    ----------
    sorted_graph : List[Tensor]
        [description]
    """
    global func_cnt
    global tensor_cnt
    global func_name_to_tensor_cnt
    for node in sorted_graph:
        funcs = node.src_ops()
        if len(funcs) == 0:
            if node._attrs["name"] is None:
                tensor_name = unique_name(f"tensor_{tensor_cnt}")
                node._attrs["name"] = tensor_name
                tensor_cnt += 1

        else:
            for func in funcs:
                if func._attrs["name"] is None:
                    func_name = "{op_kind}_{idx}".format(
                        op_kind=func._attrs["op"], idx=func_cnt
                    )
                    func_name = unique_name(func_name)
                    func._attrs["name"] = func_name
                    func._attrs["original_name"] = func_name
                    func_cnt += 1
                    func_name_to_tensor_cnt[func_name] = 0
                if node._attrs["name"] is None:
                    func_tensor_count = func_name_to_tensor_cnt[func_name]
                    node_name = unique_name(f"{func_name}_{func_tensor_count}")
                    node._attrs["name"] = node_name
                    func_name_to_tensor_cnt[func_name] = func_tensor_count + 1
        tensor_name = node._attrs["name"]
        for i, dim in enumerate(node._attrs["shape"]):
            if dim._attrs["name"] is None:
                dim_name = "{tname}_dim_{idx}".format(tname=tensor_name, idx=i)
                dim._attrs["name"] = dim_name
