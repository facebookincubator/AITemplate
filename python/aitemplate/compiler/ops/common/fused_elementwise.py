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
Fused elementwise operator definition.
"""
from typing import List, Set

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator
from aitemplate.compiler.ops.common.elementwise import elementwise
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0301,C0103,W0223


def _check_shapes_eq(shapes1, shapes2) -> bool:
    if len(shapes1) != len(shapes2):
        return False
    for shape1, shape2 in zip(shapes1, shapes2):
        if shape1 != shape2:
            return False
    return True


class fused_elementwise(Operator):
    """fused_elementwise operator is used internally.
    It's the actual operator which does ++ codegen.
    """

    def _check_output_shape(self) -> None:
        outputs = self._attrs["outputs"]
        shape = outputs[0]._attrs["shape"]
        for i in range(1, len(outputs)):
            if not _check_shapes_eq(shape, outputs[i]._attrs["shape"]):
                raise RuntimeError(
                    "Output shapes of fused_elementwise Op do not match! Shape1: {}. Shape2: {}.".format(
                        shape, outputs[i]._attrs["shape"]
                    )
                )

    def _update_inputs_outputs(
        self, inputs: Set[Operator], outputs: Set[Operator]
    ) -> None:
        self._attrs["inputs"] = list(inputs)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._attrs["outputs"] = list(outputs)
        self._attrs["output_accessors"] = [
            TensorAccessor(output_tensor) for output_tensor in self._attrs["outputs"]
        ]
        self._check_output_shape()

        # Preserve original tensors in case there are scatter / gather fusions.
        # Need to copy Tensor objects.
        self._attrs["original_inputs"] = list(self._attrs["inputs"])
        self._attrs["original_outputs"] = list(self._attrs["outputs"])

        for tensor in inputs:
            tensor._attrs["dst_ops"].add(self)
        for tensor in outputs:
            tensor._attrs["src_ops"].add(self)

    def _check_constant(self) -> None:
        if len(self._attrs["inputs"]) == 0:
            raise RuntimeError(f"No inputs for fused_elementwise! {self}")
        for input_tensor in self._attrs["inputs"]:
            if not input_tensor.is_a_const_num():
                return
        raise NotImplementedError(
            "Cannot handle the case that all inputs of a fused_elementwise are constant numbers! "
            f"Please use Python to calculate directly. Operator: {self}"
        )

    def __init__(
        self,
        elementwise_ops: List[elementwise],
        inputs: Set[Operator],
        outputs: Set[Operator],
    ) -> None:
        super().__init__()

        if len(elementwise_ops) == 0:
            raise RuntimeError(
                "fused_elementwise argument elementwise_ops cannot be empty!"
            )
        # It is required that elementwise_ops need to be topologically sorted.
        self._attrs["op"] = "fused_elementwise"
        self._attrs["elementwise_ops"] = elementwise_ops
        self._attrs["has_profiler"] = False

        self._update_inputs_outputs(inputs, outputs)
        self._set_depth()
        self._check_constant()

    def _get_op_attributes(self):
        return {
            "elementwise_ops": self._attrs["elementwise_ops"],
            "inputs": self._attrs["inputs"],
            "outputs": self._attrs["outputs"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def _args_for_pseudo_code(self):
        return [op._attrs["func"] for op in self._attrs["elementwise_ops"]]
