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
Graph pass to assign names to a sorted graph.
"""
import re
from typing import List

from ..base import IntVarTensor, Tensor

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
    """Provide each tensor and operator with a unique valid C variable name

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph to be named
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
                if isinstance(node, IntVarTensor):
                    # TODO: emit standalone dynamic shape initialization for IntVarTensor
                    raise RuntimeError(
                        "We don't support emitting standalone IntVarTensor at this moment.\n"
                        f"Encountered {node._attrs['name']}: {node._attrs['int_var']}."
                    )

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
                    if isinstance(node, IntVarTensor):
                        shape_name = node._attrs["int_var"]._attrs["name"]
                        if shape_name is None:
                            node._attrs["int_var"]._attrs["name"] = node_name

        tensor_name = node._attrs["name"]
        for i, dim in enumerate(node._attrs["shape"]):
            if dim._attrs["name"] is None:
                dim_name = "{tname}_dim_{idx}".format(tname=tensor_name, idx=i)
                dim._attrs["name"] = dim_name
