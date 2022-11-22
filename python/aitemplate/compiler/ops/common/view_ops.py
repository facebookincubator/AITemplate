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
View ops.
"""

import itertools
import logging
import math
from typing import Any, List, Optional, Tuple, Union

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor
from aitemplate.utils.shape_utils import convert_shape_to_IntVar

from ....utils.tensor_utils import wrap_dim


# SHAPE_ASSIGNMENT_TEMPLATE is folded in here
# Only used in generating C++ code
RESHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{% if unknown_idx >= 0 %}
{% for idx in range(input_ndim) %}
{{indent}}{{dtype}}IN_{{idx}} = *in_{{idx}};
{% endfor %}

{% for idx in range(output_ndim) %}
{{indent}}{{dtype}}OUT_{{idx}} = *out_{{idx}};
{% endfor %}

{{indent}}{{dtype}}prod = 1;
{% for idx in range(input_ndim) %}
{{indent}}prod *= IN_{{idx}};
{% endfor %}

{{indent}}{{dtype}}out_prod = 1;

{% for j in range(0, unknown_idx) %}
{{indent}}out_prod *= OUT_{{j}};
{% endfor %}
{% for j in range(unknown_idx + 1, output_ndim) %}
{{indent}}out_prod *= OUT_{{j}};
{% endfor %}

{{indent}}*out_{{unknown_idx}} = prod / out_prod;

{% endif %}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

DYNAMIC_RESHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{% for idx in range(input_ndim) %}
{{indent}}*out_{{idx}} = *in_{{idx}};
{% endfor %}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

SQUEEZE_FUNC_TEMPLATE = jinja2.Template(
    """
{% for idx in range(output_ndim) %}
{% if idx in out_dim_to_in %}
{{indent}}*out_{{idx}} = *in_{{out_dim_to_in[idx]}};
{% endif %}
{% endfor %}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

# no EXEC_COND_TEMPLATE because there is no cuda/rocm kernel generated for reshape

# pylint: disable=C0103,W0221,R1732,W0613
logging.basicConfig(level=logging.INFO)


class _view(Operator):
    """
    Base class for View operators.
    """

    def replace_input_tensor(self, old_tensor, new_tensor) -> None:
        super().replace_input_tensor(old_tensor, new_tensor)
        for output in self._attrs["outputs"]:
            if output._attrs["is_view_of"] is old_tensor:
                output._attrs["is_view_of"] = new_tensor


class _reshape_base(_view):
    """
    Base class for reshape and flatten
    """

    def __init__(self):
        super().__init__()
        self._attrs["unknown_idx"] = -1

    def make_output_shape(
        self,
        y_shape_values: List[Union[List[int], int]],
        dynamic_dim: IntVar = None,
        is_intvar_tensor: bool = False,
    ) -> List[IntVar]:
        """
        Make the output shape from the output shape values.
        """
        output_shape = []
        for idx, values in enumerate(y_shape_values):
            if len(values) == 1:
                output_shape.append(IntImm(values[0]))
            else:
                if not is_intvar_tensor:
                    assert (
                        self._attrs["unknown_idx"] == -1
                    ), f"{self._attrs['op']} doesn't support multiple dynamic dims, "
                    "got {idx} and {self._attrs['unknown_idx']}"
                    self._attrs["unknown_idx"] = idx
                output_shape.append(
                    dynamic_dim if dynamic_dim is not None else IntVar(values=values)
                )
        return output_shape


def _is_dynamic_dim_reused(x_shape_values, y_shape_values) -> bool:
    x_cumulative_static_dim = math.prod(v[0] for v in x_shape_values if 1 == len(v))
    y_cumulative_static_dim = math.prod(v[0] for v in y_shape_values if 1 == len(v))
    x_count_dynamic_dims = sum(len(v) > 1 for v in x_shape_values)
    y_count_dynamic_dims = sum(len(v) > 1 for v in y_shape_values)

    # if there is a single dynamic dim in current and output shape,
    # and the values for dynamic dim are same between current and output shape,
    # (equivalently, product of static dims is the same),
    # we can reuse the current dynamic dim in the output shape;
    # otherwise, a new dynamic dim will be created.
    return (
        x_count_dynamic_dims == y_count_dynamic_dims
        and x_cumulative_static_dim == y_cumulative_static_dim
        and x_count_dynamic_dims == 1
    )


class reshape(_reshape_base):
    """
    Returns a tensor with the same data and number of elements as input, but with the
    specified shape. Inputs must be contiguous.

    A single dimension may be -1, in which case it’s inferred from the remaining
    dimensions and the number of elements in input.

    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "reshape"
        self.shape_eval_template = RESHAPE_FUNC_TEMPLATE
        self.dynamic_eval_template = DYNAMIC_RESHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: Tuple[int], shape: Tuple[int]):
        new_shape = list(shape)
        cur_shape = x
        unknown_idx = -1
        prod = 1
        for idx, v in enumerate(new_shape):
            if v == -1:
                # no multiple -1s
                assert unknown_idx == -1
                unknown_idx = idx
            else:
                prod *= v
        numel = 1
        for dim in cur_shape:
            numel *= dim

        if unknown_idx == -1:
            assert (
                numel == prod
            ), f"When there is no unknown index, we expect dim products to be equal, got current shape {numel=} != new shape {prod=}"
        else:
            # FIXME: note that this RuntimeError rules out some "valid" PyTorch
            # code like:
            # t = torch.arange(0).reshape(4, 0)
            # this is valid in PT but would trigger RuntimeError below
            # t.reshape(2, 2, -1)
            # We can fix it later.
            if prod <= 0:
                raise RuntimeError(f"cannot reshape tensor {x} with shape {shape}")
            assert numel % prod == 0
            new_shape[unknown_idx] = numel // prod
        return new_shape

    def _infer_shapes(self, x: Tensor):
        # There are two cases:
        # 1) there is only one unknown shape.
        # 2) there is no unkown shape and all shape dimensions are represented as IntVarTensor
        # For 1), the view op will deduce the shape of if one dim is labeled as -1,
        #         but it can't do so with more than 1 dynamic dimension
        # For 2), when all dynamic shapes are known, we should be able to pass the input shape to out.
        #         i.e. we should skip the deduction when all shapes are known.
        is_intvar = all([isinstance(var, IntVarTensor) for var in self._attrs["shape"]])
        self._attrs["is_intvar"] = is_intvar
        if not is_intvar:
            x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
            x_dynamic_dims = [
                var for var in x._attrs["shape"] if 1 < len(var._attrs["values"])
            ]
            x_shapes = list(itertools.product(*x_shape_values))

            self._attrs["shape"] = convert_shape_to_IntVar(self._attrs["shape"])
            new_shape_vals = [var._attrs["values"] for var in self._attrs["shape"]]
            new_shapes = list(itertools.product(*new_shape_vals))

            # len(x_shapes) > 1 means that at least 1 dim in the shapes of x is dynamic.
            # len(new_shapes) > 1 means that the dynamic dim is retained; otherwise, it would
            # have been replaced with -1 or a concrete number.
            if len(x_shapes) > len(new_shapes):
                # we only support two cases here, when len(x_shapes) > 1, len(x_shapes) must
                # be either len(new_shapes) (the dynamic dim is retained) or 1 (use -1 to
                # mark the dynamic or unknown index and no other dim is dynamic).
                assert len(new_shapes) == 1
                new_shapes = new_shapes * len(x_shapes)
            # run infershape for each
            y_shapes = [
                self._infer_shape(x_shape, new_shape)
                for x_shape, new_shape in zip(x_shapes, new_shapes)
            ]

            def unique(vector):
                return sorted(set(vector))

            y_shape_values = list(map(unique, zip(*y_shapes)))
            reuse_dynamic_dim = _is_dynamic_dim_reused(x_shape_values, y_shape_values)
            return self.make_output_shape(
                y_shape_values,
                dynamic_dim=x_dynamic_dims[0] if reuse_dynamic_dim else None,
            )
        else:
            new_shape_vals = [
                shape._attrs["int_var"]._attrs["values"]
                for shape in self._attrs["shape"]
            ]
            return self.make_output_shape(new_shape_vals, is_intvar_tensor=True)

    def __call__(self, x: Tensor, shape: List[Any]) -> Tensor:
        self._attrs["shape"] = shape
        self._attrs["inputs"] = [x]
        for s in shape:
            if isinstance(s, IntVarTensor):
                # Add IntVarTensors to inputs as well.
                self._attrs["inputs"].append(s)
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, is_view_of=x)
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        # There are two cases:
        # 1) there is only one unknown shape.
        # 2) there is no unkown shape and all shape dimensions are represented as IntVarTensor
        # For 1), at implementation, the uknown dimension = X.flatten()/(*known_out_shape)
        # For 2), when all dynamic shapes are intVarTensor, output_shape = input_shape.
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        if self._attrs["is_intvar"]:
            return func(self._attrs, self.dynamic_eval_template)
        else:
            return func(self._attrs, self.shape_eval_template)

    def _inputs_for_pseudo_code(self):
        return [
            self._attrs["inputs"][0],
            f"shape=[{self._pseudo_code_helper(self._attrs['shape'], with_shape=True)}]",
        ]


class flatten(_reshape_base):
    """
    Flattens input by reshaping it into a one-dimensional tensor. If start_dim or end_dim
    are passed, only dimensions starting with start_dim and ending with end_dim are
    flattened. The order of elements in input is unchanged.
    """

    def __init__(self, start_dim=0, end_dim=-1) -> None:
        super().__init__()
        self._attrs["op"] = "flatten"
        self.shape_eval_template = RESHAPE_FUNC_TEMPLATE
        self._attrs["start"] = start_dim
        self._attrs["end"] = end_dim

    def _infer_shape(self, x: List[int]):
        start = self._attrs["start"]
        end = self._attrs["end"]

        start = wrap_dim(start, len(x))
        end = wrap_dim(end, len(x))

        new_shape = []
        for idx in range(start):
            new_shape.append(x[idx])

        prod = 1
        for dim in x[start : end + 1]:
            prod *= dim
        new_shape.append(prod)

        for dim in x[end + 1 :]:
            new_shape.append(dim)

        return new_shape

    def _infer_shapes(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        x_dynamic_dims = [
            var for var in x._attrs["shape"] if 1 < len(var._attrs["values"])
        ]

        # run infershape for each
        y_shapes = [self._infer_shape(x_shape) for x_shape in x_shapes]

        def unique(vector):
            return sorted(set(vector))

        y_shape_values = list(map(unique, zip(*y_shapes)))
        reuse_dynamic_dim = _is_dynamic_dim_reused(x_shape_values, y_shape_values)
        return self.make_output_shape(
            y_shape_values,
            dynamic_dim=x_dynamic_dims[0] if reuse_dynamic_dim else None,
        )

    def _sanity_check(self, x_shape):
        x_rank = len(x_shape)
        start_dim = wrap_dim(self._attrs["start"], x_rank)
        end_dim = wrap_dim(self._attrs["end"], x_rank)
        assert (
            start_dim >= 0 and start_dim < x_rank
        ), f"flatten start_dim={start_dim} must be non-negative and less than input rank={x_rank}"
        assert (
            end_dim >= 0 and end_dim < x_rank
        ), f"flatten end_dim={end_dim} must be non-negative and less than input rank={x_rank}"
        assert (
            start_dim <= end_dim
        ), f"flatten start_dim={start_dim} must be less than or equal to end_dim={end_dim}"

    def __call__(self, x: Tensor) -> Tensor:
        self._sanity_check(x._attrs["shape"])
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, is_view_of=x)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"start_dim": self._attrs["start"], "end_dim": self._attrs["end"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, self.shape_eval_template)

    def _args_for_pseudo_code(self):
        return [f"start={self._attrs['start']}", f"end={self._attrs['end']}"]


class squeeze(_view):
    """
    Examines the specified dimension and gets rid of it if it is of size 1.

    .. highlight:: python
    .. code-block:: python

        >>> x = Tensor(shape=[IntImm(3), IntImm(2), IntImm(1)])
        >>> squeeze(2)(x)
        Tensor(shape=[IntImm(3), IntImm(2)])

        >>> x = Tensor(shape=[IntImm(3), IntImm(2), IntImm(1)])
        >>> squeeze(1)(x)
        Tensor(shape=[IntImm(3), IntImm(2), IntImm(1)])

        >>> x = Tensor(shape=[IntImm(4), IntImm(1), IntImm(3)])
        >>> squeeze(-2)(x)
        Tensor(shape=[IntImm(4), IntImm(3)])

        >>> x = Tensor(shape=[IntImm(1), IntImm(1), IntImm(4)])
        >>> squeeze(None)(x)
        Tensor(shape=[IntImm(4)])

    There are some additional assumptions for dynamic dims. Since our shape inference
    system cannot handle outputs with variable outputs, we assume that if a dynamic dim
    is squeezed, it contains no ones:

    .. highlight:: python
    .. code-block:: python

        >>> x = Tensor(shape=[IntVar([3, 2]), IntImm(2)])
        >>> y = Tensor(shape=[IntVar([1, 2]), IntImm(2)])
        >>> squeeze(0)(x) # OK
        Tensor(shape=[IntVar([3, 2]), IntImm(2)])
        >>> squeeze(1)(y) # error!

    * :attr:`dim (Optional[int])` : the dimension to get rid of. If None, get rid of all dimensions of size 1.

    Args:
        x (Tensor): the source tensor to squeeze.

    Returns:
        Tensor: the squeezed tensor.
    """

    def __init__(self, dim: Optional[int]) -> None:
        super().__init__()
        self._attrs["op"] = "squeeze"
        self._attrs["dim"] = dim
        self.shape_eval_template = SQUEEZE_FUNC_TEMPLATE

    def _infer_shapes(self, x: Tensor) -> IntVar:
        dim = self._attrs["dim"]
        x_shape = x._attrs["shape"]

        if dim is not None:
            dim = wrap_dim(self._attrs["dim"], len(x_shape))

        new_shape = []
        out_dim_to_in = {}
        out_dim = 0
        for input_idx, shape in enumerate(x_shape):
            if (dim is None or input_idx == dim) and shape == IntImm(1):
                # This dim is squeezed
                continue

            if isinstance(shape, IntVar):
                # Dynamic shape needs to be written to in generated code.
                # Save it here.
                out_dim_to_in[out_dim] = input_idx
            out_dim += 1
            new_shape.append(shape)

        self._attrs["out_dim_to_in"] = out_dim_to_in
        return new_shape

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, is_view_of=x)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"dim": self._attrs["dim"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, self.shape_eval_template)

    def _args_for_pseudo_code(self):
        return [f"dim={self._attrs['dim']}"]


class unsqueeze(squeeze):
    """
    Adds a dimension of size 1 at a specified location.
    >>> x = Tensor(shape=[IntImm(4), IntImm(3)])
    >>> unsqueeze(0)(x)
    Tensor(shape=[IntImm(1), IntImm(4), IntImm(3)])
    >>> unsqueeze(-1)(x)
    Tensor(shape=[IntImm(4), IntImm(3), IntImm(1)])

    Args:
        dim (int): Where to add the dimension, must be in range [-input_ndim - 1, input_dim + 1)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._attrs["op"] = "unsqueeze"
        self._attrs["dim"] = dim

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        x_shape = x._attrs["shape"]
        dim = wrap_dim(self._attrs["dim"], len(x_shape) + 1)

        y_shapes = []
        out_dim_to_in = {}
        out_dim = 0
        for idx, shape in enumerate(x_shape):
            if idx == dim:
                y_shapes.append(IntImm(1))
                out_dim += 1

            if isinstance(shape, IntVar):
                out_dim_to_in[out_dim] = idx

            y_shapes.append(shape)
            out_dim += 1

        if len(y_shapes) == len(x_shape):
            # New dim is added at the end
            y_shapes.append(IntImm(1))

        self._attrs["out_dim_to_in"] = out_dim_to_in
        return y_shapes
