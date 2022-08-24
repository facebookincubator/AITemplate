# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import itertools
from typing import List

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ....utils.tensor_utils import wrap_dim
from ...base import IntImm, IntVar, Operator, Tensor

# pylint: disable=C0103,W0221


class split(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self) -> None:
        """[summary]

        Parameters
        ----------

        Returns
        """
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
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        Returns
        -------
        Tensor
            [description]
        """
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
        # torch returns a tuple, so do we
        return tuple(outputs)

    def _get_func(self, fmt_str):
        """[summary]

        Parameters
        ----------
        inputs : string
            [description] format string to create func_key for looking up func
                          from the registry

        Returns
        -------
        [type]
            [description]
        """
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_decl")
        return func(self._attrs)

    def gen_function_call(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_call")
        return func(self._attrs)

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"] + [
            f"split_sizes={str(self._attrs['split_sizes'])}]",
            f"dim={str(self._attrs['split_dim'])}]",
        ]
