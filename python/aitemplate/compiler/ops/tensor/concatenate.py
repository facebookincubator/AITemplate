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
Concatenate.
"""
from typing import List, Sequence, Union

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ....utils.tensor_utils import wrap_dim
from ...base import IntVar, Operator, Tensor
from ...tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0221


class concatenate(Operator):
    """
    Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    It is the inverse operation for `split` and `chunk`.

    Args:
        inputs (List[Tensor]): the sequence of input tensors to concatenate
        dim (int): the dimension to concatenate. Optional, 0 by default

    Returns:
        Tensor: the output tensor

    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "concatenate"
        self._attrs["has_profiler"] = False

    def _unique(self, vector):
        return sorted(set(vector))

    def _infer_shapes(self, inputs: List[Tensor], dim) -> List[IntVar]:
        """Infers shapes for concatenate."""

        if len(inputs) < 1:
            raise RuntimeError("expected a list of Tensors")
        x = inputs[0]
        rank = len(x._attrs["shape"])
        if rank <= 0:
            raise RuntimeError("expected a non-scalar tensor")
        if dim >= rank:
            raise RuntimeError(
                "concat_dim ({dim}) expected to be less than rank ({rank})".format(
                    dim=dim, rank=rank
                )
            )
        for t in inputs:
            r = len(t._attrs["shape"])
            if r != rank:
                raise RuntimeError(
                    "tensors expected to have the same rank, got {} and {}".format(
                        r, rank
                    )
                )
        input_shapes = [i._attrs["shape"] for i in inputs]
        output_shape = []
        input_shape_values = [
            [d._attrs["values"] for d in shape] for shape in input_shapes
        ]
        for idx, lst in enumerate(zip(*input_shape_values)):
            if idx == dim:
                min_value_sum = sum(value[0] for value in lst)
                max_value_sum = sum(value[-1] for value in lst)
                output_shape.append(
                    shape_utils.gen_int_var([min_value_sum, max_value_sum])
                )
            else:
                output_dim = input_shapes[0][idx]
                for shape in input_shapes:
                    if output_dim != shape[idx]:
                        raise RuntimeError(
                            "tensors expected to have the same dimensions "
                            "except concat_dim! dim: {}, shape1: {}, shape2: {}, inputs: {}".format(
                                idx, output_dim, shape[idx], inputs
                            )
                        )
                output_shape.append(output_dim)
        return output_shape

    def __call__(self, inputs: List[Tensor], dim=0) -> List[Tensor]:
        self._attrs["inputs"] = list(inputs)
        self._attrs["input_accessors"] = [
            TensorAccessor(t) for t in self._attrs["inputs"]
        ]
        # We have transformations that may modify some inputs to tensor accessors,
        # for which the source op will write directly to the corresponding
        # output locations. However, our concat backend needs original input
        # shapes to calculate concat offsets. So, we keep a copy of input tensors.
        self._attrs["original_inputs"] = list(inputs)
        # True means the corresponding tensor will be copied by the concat backend.
        self._attrs["input_masks"] = [True] * len(inputs)
        input_rank = inputs[0]._rank()
        dim = wrap_dim(dim, input_rank)
        self._attrs["concat_dim"] = dim
        self._set_depth()
        output_shape = self._infer_shapes(inputs, dim)
        output = Tensor(output_shape, src_ops={self}, dtype=inputs[0]._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_func(self, fmt_str):
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def get_original_index(self, idx: int) -> int:
        """
        Return the original index of the input at idx in the current "inputs" list.

        Parameters
        ----------
        idx : int
            the index of an input based on the current "inputs"

        Returns
        -------
        int
            the index of this input in the "original_inputs"
        """
        num_original_inputs = len(self._attrs["original_inputs"])
        orig_idx = None
        # track the index for the "inputs" list
        curr_idx = 0
        for i in range(num_original_inputs):
            # We don't increase curr_idx if this input is removed
            if not self._attrs["input_masks"][i]:
                continue
            # We found the original index
            if curr_idx == idx:
                orig_idx = i
                break
            curr_idx += 1
        assert orig_idx is not None, f"Expected orig_idx to be non-None for idx {idx}"
        return orig_idx

    def get_tensor_index(self, tensor: Tensor) -> int:
        """
        Return the index for the input tensor in the "inputs" list.

        Parameters
        ----------
        tensor : Tensor
            the input tensor for looking up the index

        Returns
        -------
        int
            the index of this input in the "nputs" list
        """
        idx = None
        for input_idx, input_tensor in enumerate(self._attrs["inputs"]):
            if input_tensor is tensor:
                idx = input_idx
                # found the input to be removed
                break
        assert idx is not None and idx < len(self._attrs["inputs"]), (
            f"Expected idx to be less than the number of inputs, "
            f'but got: {idx}, {len(self._attrs["inputs"])}'
        )
        return idx

    def remove_input_at(self, indices: Union[int, Sequence[int]]) -> None:
        """
        This function removes the inputs in indices from the "inputs" attribute
        and sets input_masks[indices] to be False. Note that the indices are based
        on the current "inputs".

        Parameters
        ----------
        indices : Union[int, Sequence[int]]
            the index of an input or indices of multiple inputs based on the current "inputs"

        Returns
        -------
        None
        """
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)

        curr_inputs = self._attrs["inputs"]
        curr_input_accessors = self._attrs["input_accessors"]
        num_curr_inputs = len(curr_inputs)

        assert len(curr_input_accessors) == num_curr_inputs, (
            "expected curr_input_accessors have the same length as num_curr_inputs, "
            f"but got {len(curr_input_accessors)=}, {num_curr_inputs=}, "
            f'op: {self._attrs["name"]}'
        )

        assert (
            len(indices) <= num_curr_inputs
        ), f"Expected len(indices) <= num_curr_inputs, but got {len(indices)} and {num_curr_inputs}"

        num_original_inputs = len(self._attrs["original_inputs"])
        num_input_masks = len(self._attrs["input_masks"])
        assert num_original_inputs == num_input_masks, (
            f"original_inputs and input_masks must have the same length, "
            f"but got {num_original_inputs} and {num_input_masks}"
        )

        curr_idx = 0  # index into curr_inputs
        idx = 0  # index into indices
        new_inputs = []
        new_input_accessors = []
        # we need to skip those indices where input_masks have been modified.
        for orig_idx in range(num_original_inputs):
            if not self._attrs["input_masks"][orig_idx]:
                continue
            if idx < len(indices) and curr_idx == indices[idx]:
                if not self._attrs["input_masks"][orig_idx]:
                    raise RuntimeError(
                        f'Expected input_masks at {idx} to be True for {self._attrs["name"]}'
                    )
                if curr_input_accessors[curr_idx].stride_dim is not None:
                    raise RuntimeError(
                        f"Cannot remove an input (idx: {curr_idx}) with a valid "
                        f'TensorAccessor for {self._attrs["name"]}'
                    )
                self._attrs["input_masks"][orig_idx] = False
                idx += 1
            else:
                new_inputs.append(curr_inputs[curr_idx])
                new_input_accessors.append(curr_input_accessors[curr_idx])
            curr_idx += 1
        num_new_inputs = len(new_inputs)
        assert num_new_inputs + len(indices) == num_curr_inputs, (
            f"Expected num_new_inputs + len(indices) == num_curr_inputs, "
            f"but got {num_new_inputs + len(indices)} and {num_curr_inputs}"
        )
        self._attrs["inputs"] = new_inputs
        self._attrs["input_accessors"] = new_input_accessors

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"] + [f"dim={self._attrs['concat_dim']}"]
