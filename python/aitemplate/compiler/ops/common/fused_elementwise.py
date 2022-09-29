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
from typing import List

from .... import backend
from ....backend import registry
from ...base import Operator
from ...tensor_accessor import TensorAccessor
from .elementwise import elementwise

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

    def _update_inputs_outputs(self) -> None:
        ops = set(self._attrs["elementwise_ops"])
        external_inputs = set()
        external_outputs = set()
        tmp_inputs = set()
        tmp_outputs = set()

        for op in ops:
            for input_tensor in op._attrs["inputs"]:
                tmp_inputs.add(input_tensor)
                if (
                    len(input_tensor._attrs["src_ops"]) == 0
                    or len(set(input_tensor._attrs["src_ops"]) - ops) > 0
                ) and (not input_tensor.is_a_const_num()):
                    external_inputs.add(input_tensor)
                assert op in input_tensor._attrs["dst_ops"]
            for output_tensor in op._attrs["outputs"]:
                tmp_outputs.add(output_tensor)
                if (
                    output_tensor._attrs["is_output"]
                    or len(output_tensor._attrs["dst_ops"] - ops) > 0
                ):
                    external_outputs.add(output_tensor)
                assert len(output_tensor._attrs["src_ops"]) == 1
                assert list(output_tensor._attrs["src_ops"])[0] == op

        assert (
            external_inputs == tmp_inputs - tmp_outputs
        ), "external_inputs: {} is not equal to tmp_inputs: {} - tmp_outputs: {}.".format(
            external_inputs, tmp_inputs, tmp_outputs
        )
        assert (
            len(tmp_outputs - tmp_inputs - external_outputs) == 0
        ), "tmp_outputs: {} - tmp_inputs: {} - external_outputs: {} is not empty.".format(
            tmp_outputs, tmp_inputs, external_outputs
        )
        assert (
            len(external_outputs - tmp_outputs) == 0
        ), "external_outputs: {} - tmp_outputs: {} is not empty.".format(
            external_outputs, tmp_outputs
        )

        self._attrs["inputs"] = list(external_inputs)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._attrs["outputs"] = list(external_outputs)
        self._attrs["output_accessors"] = [
            TensorAccessor(output_tensor) for output_tensor in self._attrs["outputs"]
        ]
        self._check_output_shape()

        # Preserve original tensors in case there are scatter / gather fusions.
        # Need to copy Tensor objects.
        self._attrs["original_inputs"] = list(self._attrs["inputs"])
        self._attrs["original_outputs"] = list(self._attrs["outputs"])

        for tensor in tmp_inputs | tmp_outputs:
            tensor._attrs["src_ops"] = set(tensor._attrs["src_ops"]) - ops
            tensor._attrs["dst_ops"] = tensor._attrs["dst_ops"] - ops
        for tensor in external_inputs:
            tensor._attrs["dst_ops"].add(self)
        for tensor in external_outputs:
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

    def __init__(self, elementwise_ops: List[elementwise]) -> None:
        super().__init__()

        if len(elementwise_ops) == 0:
            raise RuntimeError(
                "fused_elementwise argument elementwise_ops cannot be empty!"
            )

        self._attrs["op"] = "fused_elementwise"
        self._attrs["elementwise_ops"] = elementwise_ops
        self._attrs["has_profiler"] = False

        self._update_inputs_outputs()
        self._set_depth()
        self._check_constant()

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def _args_for_pseudo_code(self):
        return [op._attrs["func"] for op in self._attrs["elementwise_ops"]]
