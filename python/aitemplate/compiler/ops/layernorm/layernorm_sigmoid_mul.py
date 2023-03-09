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
Operator definition for layernorm_sigmoid_mul.
"""
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator
from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0221,W0102,W0223


class layernorm_sigmoid_mul(Operator):
    """Fused layernorm_sigmoid_mul op
    Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
    Normalized_shape: [N1, N2, ..., ND]
    Gamma/Beta, if not None, have the same shape as normalized_shape.
    """

    def __init__(self, layer_norm: Operator, sigmoid: Operator, mul: Operator) -> None:
        super().__init__()
        self._attrs["op"] = "layernorm_sigmoid_mul"
        self._attrs["has_profiler"] = False

        assert layernorm_sigmoid_mul.is_valid(layer_norm, sigmoid, mul)
        self._update_inputs_outputs(layer_norm, sigmoid, mul)
        self._set_depth()

    @staticmethod
    def is_valid(layer_norm: Operator, sigmoid: Operator, mul: Operator) -> bool:
        if sigmoid._attrs["inputs"][0] != layer_norm._attrs["outputs"][0]:
            return False
        if len(mul._attrs["inputs"]) != 2:
            return False
        return (
            mul._attrs["inputs"][0] == sigmoid._attrs["outputs"][0]
            and mul._attrs["inputs"][1] == layer_norm._attrs["inputs"][0]
        ) or (
            mul._attrs["inputs"][1] == sigmoid._attrs["outputs"][0]
            and mul._attrs["inputs"][0] == layer_norm._attrs["inputs"][0]
        )

    def _update_inputs_outputs(self, layer_norm, sigmoid, mul):
        self._attrs["inputs"] = layer_norm._attrs["inputs"]
        self._attrs["gamma_constant"] = layer_norm._attrs["gamma_constant"]
        self._attrs["beta_constant"] = layer_norm._attrs["beta_constant"]
        self._attrs["normalized_shape"] = layer_norm._attrs["normalized_shape"]
        self._attrs["eps"] = layer_norm._attrs["eps"]
        self._attrs["outputs"] = mul._attrs["outputs"]
        self._attrs["output_accessors"] = [
            TensorAccessor(output_tensor) for output_tensor in self._attrs["outputs"]
        ]
        self._attrs["input_accessors"] = [TensorAccessor(self._attrs["inputs"][0])]

        for input_tensor in self._attrs["inputs"]:
            input_tensor._attrs["dst_ops"].discard(layer_norm)
            input_tensor._attrs["dst_ops"].discard(mul)
            input_tensor._attrs["dst_ops"].add(self)

        assert len(self._attrs["outputs"]) == 1
        output_tensor = self._attrs["outputs"][0]
        output_tensor._attrs["src_ops"] = StableSet([self])

        # update output tensor shape
        # hack for fixing dynamic shape with elementwise fusion issue
        x = self._attrs["inputs"][0]
        for i, shape_var in enumerate(output_tensor._attrs["shape"]):
            shape_var._attrs["values"] = x._attrs["shape"][i]._attrs["values"]

        sigmoid._attrs["inputs"][0]._attrs["src_ops"] = StableSet()
        sigmoid._attrs["inputs"][0]._attrs["dst_ops"] = StableSet()
        sigmoid._attrs["outputs"][0]._attrs["src_ops"] = StableSet()
        sigmoid._attrs["outputs"][0]._attrs["dst_ops"] = StableSet()

    def __call__(self):
        return self._attrs["outputs"][0]

    def _get_op_attributes(self):
        raise NotImplementedError(
            "layernorm_sigmoid_mul get op attribute not implemented"
        )

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
