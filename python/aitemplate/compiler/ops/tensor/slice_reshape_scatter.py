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
Slice_reshape_scatter.
"""

from .... import backend
from ....backend import registry
from ...base import IntImm, IntVar, Operator
from ...stable_set import StableSet

# pylint: disable=C0103,C0415,W0221


class slice_reshape_scatter(Operator):
    """represent slice + concat + reshape + concat pattern with
    slice + concat
    """

    @staticmethod
    def is_valid(cat_op: Operator, reshape_op: Operator, cat_op_2: Operator) -> bool:
        assert cat_op._attrs["op"] == "concatenate"
        assert reshape_op._attrs["op"] == "reshape"
        assert cat_op_2._attrs["op"].startswith("concatenate")

        # only handle cases where two cat ops have the same concat_dim
        cat_dim = cat_op._attrs["concat_dim"]
        if cat_dim != cat_op_2._attrs["concat_dim"]:
            return False
        cat_output_shape = cat_op._attrs["outputs"][0]._attrs["shape"]
        cat_output_rank = len(cat_output_shape)
        if cat_output_rank <= 1:
            return False
        cat_output_shape_2 = cat_op_2._attrs["outputs"][0]._attrs["shape"]
        cat_output_rank_2 = len(cat_output_shape_2)
        # only handle cases where we are concatenating the last dim
        if cat_dim != cat_output_rank - 1:
            return False
        if cat_output_rank >= cat_output_rank_2:
            return False
        if not all(
            d1._attrs["values"][0] == d2._attrs["values"][0]
            for (d1, d2) in zip(
                cat_output_shape[:cat_dim], cat_output_shape_2[:cat_dim]
            )
        ):
            return False

        reshape_to_shape = reshape_op._attrs["outputs"][0]._attrs["shape"]
        # skip dynamic shape
        if not all(isinstance(d, (IntImm, IntVar)) for d in reshape_to_shape):
            return False

        if not all(
            d1._attrs["values"][0] == d2._attrs["values"][0]
            for (d1, d2) in zip(cat_output_shape[:cat_dim], reshape_to_shape[:cat_dim])
        ):
            return False

        return all(
            x._attrs["src_ops"] is not None
            and len(x._attrs["src_ops"]) == 1
            and list(x._attrs["src_ops"])[0]._attrs["op"] == "dynamic_slice"
            for x in cat_op._attrs["inputs"]
        )

    def _update_inputs_outputs(self, cat_op, reshape_op, cat_op_2):
        from ...transform import transform_utils

        idx = -1
        for i, input_tensor in enumerate(cat_op_2._attrs["inputs"]):
            if input_tensor == reshape_op._attrs["outputs"][0]:
                idx = i
                break
        assert idx >= 0
        cat_op_2.remove_input_at(idx)
        transform_utils.remove_single_tensor_op_from_sorted_graph(reshape_op)

        self._attrs["inputs"] = [
            op._attrs["inputs"][0] for op in self._attrs["slice_ops"]
        ]
        self._attrs["outputs"] = cat_op_2._attrs["outputs"]
        for x in self._attrs["inputs"]:
            x._attrs["dst_ops"] = {self}
        for y in self._attrs["outputs"]:
            y._attrs["src_ops"].add(self)

        for op in self._attrs["slice_ops"]:
            op._attrs["outputs"][0]._attrs["src_ops"] = StableSet()
            op._attrs["outputs"][0]._attrs["dst_ops"] = StableSet()

        for x in cat_op._attrs["inputs"]:
            x._attrs["src_ops"] = StableSet()
            x._attrs["dst_ops"] = StableSet()
        for y in cat_op._attrs["outputs"]:
            y._attrs["src_ops"] = StableSet()
            y._attrs["dst_ops"] = StableSet()

    def __init__(
        self, cat_op: Operator, reshape_op: Operator, cat_op_2: Operator
    ) -> None:
        super().__init__()
        if cat_op_2._attrs["op"] == "concatenate_tanh":
            self._attrs["element_func"] = "fast_tanh"
        else:
            self._attrs["element_func"] = None
        assert slice_reshape_scatter.is_valid(cat_op, reshape_op, cat_op_2)

        self._attrs["op"] = "slice_reshape_scatter"
        self._attrs["has_profiler"] = False
        self._attrs["scatter_dim"] = cat_op._attrs["concat_dim"]
        slice_ops = []
        for x in cat_op._attrs["inputs"]:
            src_ops = x.src_ops()
            assert len(src_ops) == 1
            slice_op = list(src_ops)[0]
            slice_ops.append(slice_op)
        self._attrs["slice_ops"] = slice_ops

        self._update_inputs_outputs(cat_op, reshape_op, cat_op_2)
        self._set_depth()

    def __call__(self):
        raise RuntimeError("op {} cannot be called directly".format(self._attrs["op"]))

    def _get_func(self, fmt_str):
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs, self._attrs["element_func"])
