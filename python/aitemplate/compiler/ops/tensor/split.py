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
Split.
"""
import itertools
from typing import List, Sequence, Union

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.utils import shape_utils
from aitemplate.utils.tensor_utils import wrap_dim

# pylint: disable=C0103,W0221


class split(Operator):
    """Splits the tensor into chunks on the specified dimension.

    Args:
        x (Tensor): tensor to split.
        split_sizes (List[int]) : list of sizes for each chunk
        dim (int): dimension along which to split the tensor

    Returns:
        List[Tensor]: the list of output tensors

    Example:

        .. highlight:: python
        .. code-block:: python

            >>> X = Tensor(shape=[2, 1], name="X", is_input=True)
            >>> Y = ops.split()(X, 2, dim=0)
            [Tensor(shape=[IntImm(1), IntImm(1)]), Tensor(shape=[IntImm(1), IntImm(1)])]
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "split"
        self._attrs["has_profiler"] = False

    def _infer_shapes(
        self, x: Tensor, split_sizes: List[int], dim: int
    ) -> List[IntVar]:
        """Infers shapes for split."""

        x_shape = x._attrs["shape"]
        rank = len(x_shape)
        if rank <= 0:
            raise RuntimeError("expected a non-scalar tensor")
        if dim >= rank:
            raise RuntimeError(
                f"split dim ({dim}) expected to be less than rank ({rank})"
            )
        num_splits = len(split_sizes)
        if num_splits < 1:
            raise RuntimeError(
                f"the number of splits expected >=0 but got {num_splits}"
            )
        split_dim_size = x_shape[dim]._attrs["values"][0]
        if sum(split_sizes) != split_dim_size:
            raise RuntimeError(
                f"sum of split_sizes ({split_sizes}) does not match split_dim_size ({split_dim_size})"
            )

        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        y_shapes = []
        for x_shape_vals in x_shapes:
            y_shape = [list(x_shape_vals) for _ in range(num_splits)]
            for split_size, shape in zip(split_sizes, y_shape):
                shape[dim] = split_size
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shapes = []
        for idx, shapes in enumerate(zip(*y_shapes)):
            assert all(split_sizes[idx] == dims[dim] for dims in shapes)
            output_shape = []
            for i in range(len(shapes[0])):
                dim_vals = unique(dims[i] for dims in shapes)
                # propagate the name of each non-split-dim dynamic axis, which
                # may be used later by some shape checks.
                if i != dim:
                    new_dim_val = shape_utils.gen_int_var(
                        dim_vals, x_shape[i]._attrs["name"]
                    )
                else:
                    # FIXME: we might want to create a new unique name for this
                    # new_dim_val. We would do this once we have a mechanism
                    # to create a unique dim name
                    new_dim_val = shape_utils.gen_int_var(dim_vals)
                output_shape.append(new_dim_val)
            output_shapes.append(output_shape)
        return output_shapes

    def __call__(self, x: Tensor, split_size_or_sections, dim=0) -> List[Tensor]:
        x_shape = x._attrs["shape"]
        self._attrs["inputs"] = [x]
        dim = wrap_dim(dim, x._rank())
        self._attrs["split_dim"] = dim
        self._set_depth()
        if isinstance(split_size_or_sections, (List, tuple)):
            split_sizes = list(split_size_or_sections)
        else:
            split_size = split_size_or_sections
            if not isinstance(split_size, int):
                raise RuntimeError("split_size expected to be of int")
            # TODO: support split along dynamic axis
            if not isinstance(x_shape[dim], IntImm):
                raise NotImplementedError("split dynamic axis")
            split_dim_size = x_shape[dim].value()
            if split_dim_size == 0:
                # a special case - it's valid in pytorch
                num_splits = 1
                split_sizes = [0]
            else:
                if split_size == 0:
                    raise RuntimeError("split_size expected to be > 0")
                num_splits = int((split_dim_size + split_size - 1) / split_size)
                split_sizes = [split_size] * num_splits
                split_sizes[num_splits - 1] = split_size - (
                    split_size * num_splits - split_dim_size
                )

        self._attrs["split_sizes"] = split_sizes
        output_shapes = self._infer_shapes(x, split_sizes, dim)
        outputs = [
            Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
            for output_shape in output_shapes
        ]
        self._attrs["outputs"] = outputs
        self._attrs["original_outputs"] = list(outputs)
        # True means the corresponding output tensor will be materialized by backend.
        self._attrs["output_masks"] = [True] * len(outputs)
        # torch returns a tuple, so do we
        return tuple(outputs)

    def _get_func(self, fmt_str):
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def remove_output_at(self, indices: Union[int, Sequence[int]]) -> None:
        """
        This function removes the outputs in indices from the "outputs" attribute
        and sets output_masks[indices] to be False. Note that the indices are based
        on the current "outputs".

        Parameters
        ----------
        indices : Union[int, Sequence[int]]
            the index of an output or indices of multiple outputs based on the current "outputs"

        Returns
        -------
        None
        """
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)

        curr_outputs = self._attrs["outputs"]
        num_curr_outputs = len(curr_outputs)

        assert (
            len(indices) <= num_curr_outputs
        ), f"Expected len(indices) <= num_curr_outputs, but got {len(indices)} and {num_curr_outputs}"

        num_original_outputs = len(self._attrs["original_outputs"])
        num_output_masks = len(self._attrs["output_masks"])
        assert num_original_outputs == num_output_masks, (
            f"original_outputs and output_masks must have the same length, "
            f"but got {num_original_outputs} and {num_output_masks}"
        )

        curr_idx = 0  # index into curr_outputs
        idx = 0  # index into indices
        new_outputs = []
        # we need to skip those indices where output_masks have been modified.
        for orig_idx in range(num_original_outputs):
            if not self._attrs["output_masks"][orig_idx]:
                continue
            if idx < len(indices) and curr_idx == indices[idx]:
                if not self._attrs["output_masks"][orig_idx]:
                    raise RuntimeError(
                        f'Expected input_masks at {idx} to be True for {self._attrs["name"]}'
                    )
                self._attrs["output_masks"][orig_idx] = False
                idx += 1
            else:
                new_outputs.append(curr_outputs[curr_idx])
            curr_idx += 1
        num_new_outputs = len(new_outputs)
        assert num_new_outputs + len(indices) == num_curr_outputs, (
            f"Expected num_new_outputs + len(indices) == num_curr_outputs, "
            f"but got {num_new_outputs + len(indices)} and {num_curr_outputs}"
        )
        self._attrs["outputs"] = new_outputs

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"] + [
            f"split_sizes={str(self._attrs['split_sizes'])}]",
            f"dim={str(self._attrs['split_dim'])}]",
        ]
