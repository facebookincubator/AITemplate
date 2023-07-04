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
Slice_scatter.
"""

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator
from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0221


class slice_scatter(Operator):
    """This op represents a special fusion case where the
    inputs of a concatenate op all come from slice ops. In such a case,
    we can remove the concatenate op by placing each slice's output
    into the correct location in the original concatenate's output.
    """

    @staticmethod
    def is_valid(cat_op: Operator) -> bool:
        if cat_op._attrs["op"] != "concatenate":
            return False
        if any(
            input_accessor.stride_dim is not None
            for input_accessor in cat_op._attrs["input_accessors"]
        ):
            return False
        return all(
            x._attrs["src_ops"] is not None
            and len(x._attrs["src_ops"]) == 1
            and len(x._attrs["dst_ops"]) == 1
            and list(x._attrs["src_ops"])[0]._attrs["op"] == "dynamic_slice"
            for x in cat_op._attrs["inputs"]
        )

    def _update_inputs_outputs(self, cat_op):
        self._attrs["inputs"] = []
        for slice_op in self._attrs["slice_ops"]:
            assert (
                len(slice_op._attrs["inputs"]) == 1
            ), "Slice op should only have 1 input! op: {}".format(slice_op)
            input_tensor = slice_op._attrs["inputs"][0]
            # A slice op's output may be fed into the same cat op multiple
            # times, so we make sure it's removed from the set only once.
            if slice_op in input_tensor._attrs["dst_ops"]:
                input_tensor._attrs["dst_ops"].remove(slice_op)
                input_tensor._attrs["dst_ops"].add(self)
            self._attrs["inputs"].append(input_tensor)

        # The original output of this slice_scatter op is the output of the cat_op.
        # We set the TensorAccessor, but will only use its offset field in the backend.
        self._attrs["output_accessors"] = [TensorAccessor(cat_op._attrs["outputs"][0])]

        self._attrs["outputs"] = cat_op._attrs["outputs"]
        for y in self._attrs["outputs"]:
            y._attrs["src_ops"] = StableSet({self})

        for op in self._attrs["slice_ops"]:
            op._attrs["outputs"][0]._attrs["src_ops"] = StableSet()
            op._attrs["outputs"][0]._attrs["dst_ops"] = StableSet()

        for x in cat_op._attrs["inputs"]:
            x._attrs["src_ops"] = StableSet()
            x._attrs["dst_ops"] = StableSet()

    def __init__(self, scatter_dim: int) -> None:
        super().__init__()
        self._attrs["op"] = "slice_scatter"
        self._attrs["has_profiler"] = False
        self._attrs["scatter_dim"] = scatter_dim

    @staticmethod
    def make_op(cat_op: Operator) -> Operator:
        assert slice_scatter.is_valid(cat_op)
        scatter_dim = cat_op._attrs["concat_dim"]
        new_op = slice_scatter(scatter_dim)
        slice_ops = []
        for x in cat_op._attrs["inputs"]:
            src_ops = x.src_ops()
            assert len(src_ops) == 1
            slice_op = list(src_ops)[0]
            slice_ops.append(slice_op)
        new_op._attrs["slice_ops"] = slice_ops
        new_op._update_inputs_outputs(cat_op)
        new_op._set_depth()
        return new_op

    def __call__(self):
        raise RuntimeError("op {} cannot be called directly".format(self._attrs["op"]))

    def _get_op_attributes(self):
        raise NotImplementedError("slice_scatter get op attribute not implemented")

    def _get_func(self, fmt_str):
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def _args_for_pseudo_code(self):
        return [f"scatter_dim={str(self._attrs['scatter_dim'])}]"]
