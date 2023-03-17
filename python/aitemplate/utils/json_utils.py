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

import json

from typing import Dict, List

from aitemplate.compiler.base import (
    _HostConstantTensorData,
    _NumpyConstantTensorData,
    _TorchConstantTensorData,
    IntImm,
    IntVar,
    Operator,
    Tensor,
)
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.tensor_accessor import TensorAccessor


def gen_unique_op_names(sorted_graph: List[Tensor]) -> Dict[Operator, str]:
    # List is used here, not Set, in order to maintain the order of operators,
    # depending on memory locations, which may vary from run to run.
    # Additionally, I don't expect to have too usages for a single op.
    tmp: Dict[str, List[Operator]] = {}
    for tensor in sorted_graph:
        for src_op in tensor.src_ops():
            op_name = src_op._attrs["name"]
            if op_name is None:
                continue

            if op_name not in tmp:
                tmp[op_name] = []
            sub_dict = tmp[op_name]

            if src_op not in sub_dict:
                sub_dict.append(src_op)

        for dst_op in tensor.dst_ops():
            op_name = dst_op._attrs["name"]
            if op_name is None:
                continue

            if op_name not in tmp:
                tmp[op_name] = []
            sub_dict = tmp[op_name]

            if dst_op not in sub_dict:
                sub_dict.append(dst_op)

    # assemble the result
    op_names: Dict[Operator, str] = {}

    for op_name, ops in tmp.items():
        if len(ops) == 1:
            # the provided operator is unique, do not add one to the dict
            continue

        # add several unique names
        for idx, op in enumerate(ops):
            op_names[op] = f"{op_name} {idx}"

    # done
    return op_names


class GraphJsonEncoder(json.JSONEncoder):
    def __init__(self, op_names: Dict[Operator, str], *args, **kwargs):
        super(GraphJsonEncoder, self).__init__(*args, **kwargs)

        # This is a Dict that provides custom names for operators.
        # It is possible that two instances of the same operator,
        # say, 'fused_elementwise_123' is used twice in the graph,
        # but with different inputs and/or outputs.
        # As a result, there will be two instances of Operator object,
        # holding the same name, which leads to invalid graph
        # visualization / serialization.
        # So, this diff allows to overcome this problem.
        self.op_names: Dict[Operator, str] = op_names

    def default(self, obj):
        if isinstance(obj, FuncEnum):
            return obj.name
        if isinstance(obj, Tensor):
            return self._jsonize_tensor(obj)
        if isinstance(obj, Operator):
            return self._jsonize_operator(obj)
        if isinstance(obj, TensorAccessor):
            return obj.__dict__
        if isinstance(obj, IntImm):
            return obj.__dict__
        if isinstance(obj, IntVar):
            return obj.__dict__
        if isinstance(obj, _HostConstantTensorData):
            return "_HostConstantTensorData"
        if isinstance(obj, _TorchConstantTensorData):
            return "_TorchConstantTensorData"
        if isinstance(obj, _NumpyConstantTensorData):
            return "_NumpyConstantTensorData"

        return str(obj)

    def _jsonize_tensor(self, tensor: Tensor):
        output = {}
        for key in tensor._attrs.keys():
            if key in ("src_ops", "dst_ops") and tensor._attrs[key] is not None:
                op_names = []
                for op in tensor._attrs[key]:
                    # check whether a name for an op is provided
                    op_name = self.op_names.get(op, op._attrs["name"])
                    op_names.append(op_name)

                output[key] = op_names
            else:
                output[key] = tensor._attrs[key]
        return output

    def _jsonize_operator(self, op: Operator):
        output = {}
        for key in op._attrs.keys():
            if (
                key in ("inputs", "args", "outputs", "original_inputs")
                and op._attrs[key] is not None
            ):
                output[key] = [x._attrs["name"] for x in op._attrs[key]]
            elif key == "name":
                # check whether a name for an op is provided.

                # save the original name
                op_name = op._attrs[key]
                output["_original_op_name"] = op_name
                # save the key
                op_name = self.op_names.get(op, op._attrs[key])
                output[key] = op_name
            else:
                output[key] = op._attrs[key]
        return output
