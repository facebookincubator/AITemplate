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


class GraphJsonEncoder(json.JSONEncoder):
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
                output[key] = [x._attrs["name"] for x in tensor._attrs[key]]
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
            else:
                output[key] = op._attrs[key]
        return output
