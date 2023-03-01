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

from ..base import IntImm, IntVarTensor, JaggedIntVar, Tensor

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
                    if not isinstance(node._attrs["int_var"], IntImm):
                        # TODO: emit standalone dynamic shape initialization for IntVarTensor
                        raise RuntimeError(
                            "We don't support emitting standalone IntVarTensor at this moment.\n"
                            f"Encountered {node._attrs['name']}: {node._attrs['int_var']}."
                        )
                    else:
                        node._attrs["int_var"]._attrs["name"] = tensor_name

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
            if dim._attrs["name"] is None and not isinstance(dim, JaggedIntVar):
                dim_name = "{tname}_dim_{idx}".format(tname=tensor_name, idx=i)
                dim._attrs["name"] = dim_name

    dim_names_in_shapes = set()
    for tensor in sorted_graph:
        for dim in tensor._attrs["shape"]:
            dim_names_in_shapes.add(dim._attrs["name"])

    for tensor in sorted_graph:
        if tensor.is_jagged():
            jagged_int_var = tensor._attrs["shape"][0]
            # JaggedIntVar's name must be the same as the name of the total_length IntVar
            # that it is based on. Due to the fact that IntVar's _attrs["name"] is accessed
            # directly throughout the code, we can't enforce this constrain by overloading
            # the name in the JaggedIntVar class. as a result, we must resort to a hack here
            # to reset the name of the JaggedIntVar to the name of the total_length after
            # the latter might have been changed (e.g., from None) by the code above.
            # TODO: wrap _attrs["name"] (and other frequently used _attrs members) in
            # @properties and override the "name" property in the JaggedIntVar to return
            # total_length().name.
            jagged_int_var._attrs["name"] = jagged_int_var.total_length()._attrs["name"]

            batch_dim = jagged_int_var.batch_dim()
            if batch_dim._attrs["name"] not in dim_names_in_shapes:
                # The batch_dim set inside the jagged_int_var is not present in any other
                # Tensor's shape directly. We mark it as isolated batch dim here to set
                # the dim to "offsets.length[0] - 1" in the make_jagged backend code.
                batch_dim._attrs["isolated"] = True
                if batch_dim._attrs["name"] is None:
                    # the batch_dim wasn't named above, so we name it here
                    jagged_int_var_name = jagged_int_var._attrs["name"]
                    batch_dim._attrs["name"] = f"{jagged_int_var_name}_jagged_batch_dim"
