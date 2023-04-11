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

import logging
import math
from functools import reduce
from typing import Any, List, Optional, Union

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import (
    IntImm,
    IntVar,
    IntVarTensor,
    JaggedDim,
    JaggedIntVar,
    Operator,
    Tensor,
)
from aitemplate.compiler.symbolic import (
    get_global_symbol_set,
    get_intvar,
    is_integer,
    is_symbol,
    is_symbolic,
    simplify_intvar_values,
)
from aitemplate.utils.shape_utils import convert_shape_to_IntVar, gen_int_var_min_max
from aitemplate.utils.tensor_utils import wrap_dim


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

    def make_output_shape_from_int_vars(
        self,
        shape: List[Any],
    ) -> List[IntVar]:
        output_shape = []
        for dim in shape:
            int_var = dim._attrs["int_var"]
            assert (
                int_var is not None
            ), f"expected an int_var dimension, but got {int_var=} for {shape=}"
            dim_values = list(int_var._attrs["values"])
            if len(dim_values) == 1:
                output_shape.append(IntImm(dim_values[0]))
            else:
                # dynamic dimension
                dim_name = int_var._attrs["name"]
                var = IntVar(
                    name=dim_name,
                    values=dim_values,
                    symbolic_value=int_var._attrs["symbolic_value"],
                )
                output_shape.append(var)
        return output_shape

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


def _get_shape_values(symbolic_shape_values, shape_values):
    new_shape_values = []
    for sym, var in zip(symbolic_shape_values, shape_values):
        if is_integer(sym):
            new_shape_values.append([int(sym)])
        else:
            new_shape_values.append(var._attrs["values"])
    return new_shape_values


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

    def _infer_shapes(self, x: Tensor):
        # There are two cases:
        # 1) there is only one unknown shape.
        # 2) there is no unkown shape and all shape dimensions are represented as IntVarTensor
        # For 1), the view op will deduce the shape of the dim that is labeled as -1,
        #         but it can't do so with more than 1 dynamic dimension
        # For 2), when all dynamic shapes are known, we should be able to pass the input shape to out.
        #         i.e. we should skip the deduction when all shapes are known.
        is_intvar = all([isinstance(var, IntVarTensor) for var in self._attrs["shape"]])
        self._attrs["is_intvar"] = is_intvar

        if not is_intvar:
            # x_symbolic_shapes is a list of symbolic_values
            x_symbolic_shapes = [
                var._attrs["symbolic_value"] for var in x._attrs["shape"]
            ]
            x_symbolic_shapes_mapping = {
                var._attrs["symbolic_value"]: var for var in x._attrs["shape"]
            }
            # x_shape_values is a list of valid IntVar _attrs["values"]
            x_shape_values = _get_shape_values(x_symbolic_shapes, x._attrs["shape"])
            # x_shape_symbolic_values is a list of valid _attrs["symbolic_values"]
            x_shape_symbolic_values = [
                shape_values[0] if len(shape_values) < 2 else sym
                for sym, shape_values in zip(x_symbolic_shapes, x_shape_values)
            ]

            self._attrs["shape"] = convert_shape_to_IntVar(self._attrs["shape"])
            to_symbolic_shapes = [
                var._attrs["symbolic_value"] for var in self._attrs["shape"]
            ]
            # new_shape_values is a list of valid IntVar _attrs["values"] with the
            # only exception being it including an -1.
            new_shape_values = _get_shape_values(
                to_symbolic_shapes, self._attrs["shape"]
            )
            new_shape_symbolic_values = [
                shape_values[0] if len(shape_values) < 2 else sym
                for sym, shape_values in zip(to_symbolic_shapes, new_shape_values)
            ]

            # Check whether we have -1 that needs to be deduced
            neg_dim = None
            for idx, s in enumerate(new_shape_values):
                if len(s) == 1 and s[0] == -1:
                    assert neg_dim is None, "Multiple -1 detected in reshape"
                    neg_dim = idx

            x_prod = reduce(
                lambda x, y: x * y, [val for val in x_shape_symbolic_values if val != 0]
            )
            new_prod = reduce(
                lambda x, y: x * y,
                [val for val in new_shape_symbolic_values if val != 0],
            )
            quotient = x_prod / new_prod
            if neg_dim is not None and is_integer(quotient):
                # We check whether the negative -1 is static.
                val = int(quotient * -1)
                new_shape_symbolic_values[neg_dim] = val
                self._attrs["shape"][neg_dim] = IntImm(val)
                neg_dim = None

            if neg_dim is None:
                # We try to simplify symbols before returning the shapes.
                symbol_idx = [
                    idx
                    for idx, s in enumerate(new_shape_symbolic_values)
                    if is_symbolic(s)
                ]

                if len(symbol_idx) == 1:
                    # Check if we can reuse shapes and if the shape belongs to
                    # unknown_idx and need to be determined during runtime.
                    new_prod = 1
                    for idx, val in enumerate(new_shape_symbolic_values):
                        if idx == symbol_idx[0]:
                            continue
                        if val != 0:
                            new_prod *= val
                    dynamic_symbol = x_prod / new_prod
                    if is_symbol(dynamic_symbol):
                        self._attrs["shape"][symbol_idx[0]] = get_intvar(
                            dynamic_symbol.name
                        )
                    elif is_integer(dynamic_symbol):
                        self._attrs["shape"][symbol_idx[0]] = IntImm(
                            int(dynamic_symbol)
                        )
                    else:
                        self._attrs["unknown_idx"] = symbol_idx[0]
                # TODO: Handle len(symbol_idx) > 1 with recording previous symbols.

                return self._attrs["shape"]
            else:
                # We try to deduce the dynamic dimensions for new_shapes.
                self._attrs["unknown_idx"] = neg_dim

                y_shapes = []
                for idx, val in enumerate(new_shape_symbolic_values):
                    if idx == self._attrs["unknown_idx"]:
                        dynamic_symbol = x_prod / new_prod * -1
                        if is_symbol(dynamic_symbol):
                            y_shapes.append(get_intvar(dynamic_symbol.name))
                        elif is_integer(dynamic_symbol):
                            y_shapes.append(IntImm(int(dynamic_symbol)))
                        else:
                            symbol_names = {s.name for s in dynamic_symbol.free_symbols}
                            assert (
                                len(symbol_names - get_global_symbol_set()) == 0
                            ), "Unable to deduce dynamic symbol"

                            values = simplify_intvar_values(dynamic_symbol)
                            new_var = IntVar(values, symbolic_value=dynamic_symbol)

                            y_shapes.append(new_var)
                    elif isinstance(val, int):
                        y_shapes.append(IntImm(val))
                    elif val in x_symbolic_shapes_mapping:
                        y_shapes.append(x_symbolic_shapes_mapping[val])
                    elif is_symbolic(val):
                        val_var = gen_int_var_min_max(
                            new_shape_values[idx], symbolic_value=val
                        )
                        y_shapes.append(val_var)
                    else:
                        raise ValueError(f"Unknown sym type for handling {val}")
            return y_shapes

        else:
            return self.make_output_shape_from_int_vars(self._attrs["shape"])

    def __call__(self, x: Tensor, shape: List[Any]) -> Tensor:
        self._attrs["shape"] = shape
        self._attrs["inputs"] = [x]
        for s in shape:
            if isinstance(s, IntVarTensor):
                # Add IntVarTensors to inputs as well.
                self._attrs["inputs"].append(s)
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(
            output_shape, src_ops={self}, is_view_of=x, dtype=x._attrs["dtype"]
        )
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

    def _infer_shapes(self, x: Tensor):
        # x_symbolic_shapes is a list of symbolic_values
        x_symbolic_shapes = [var._attrs["symbolic_value"] for var in x._attrs["shape"]]
        # x_shape_values is a list of valid IntVar _attrs["values"]
        x_shape_values = _get_shape_values(x_symbolic_shapes, x._attrs["shape"])
        # x_shape_symbolic_values is a list of valid _attrs["symbolic_values"]
        x_shape_symbolic_values = [
            shape_values[0] if len(shape_values) < 2 else sym
            for sym, shape_values in zip(x_symbolic_shapes, x_shape_values)
        ]

        start = wrap_dim(self._attrs["start"], len(x_symbolic_shapes))
        end = wrap_dim(self._attrs["end"], len(x_symbolic_shapes))
        self._attrs["unknown_idx"] = start

        # Computed shape after flatten.
        new_shapes = []

        for var in x._attrs["shape"][:start]:
            new_shapes.append(var)

        min_val, max_val, sym_val = 1, 1, 1
        for idx in range(start, end + 1):
            min_val *= min(x_shape_values[idx])
            max_val *= max(x_shape_values[idx])
            sym_val *= x_shape_symbolic_values[idx]
        if min_val == max_val:
            flatten_shape = IntImm(value=min_val)
        else:
            flatten_shape = IntVar(values=[min_val, max_val], symbolic_value=sym_val)
        new_shapes.append(flatten_shape)

        for var in x._attrs["shape"][end + 1 :]:
            new_shapes.append(var)

        return new_shapes

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
        output = Tensor(
            output_shape, src_ops={self}, is_view_of=x, dtype=x._attrs["dtype"]
        )
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


class make_jagged(_view):
    """
    Creates jagged Tensors from normal Tensors, offsets, and metadata.

    Jagged Tensors are normal Tensors with the first dynamic dimensions
    represented with a JaggedIntVar instance (as opposed to a vanilla
    IntVar). The purpose of this op is to take a normal AIT Tensor "source"
    that contains the jagged Tensor's data and return a jagged Tensor with
    the same data as source (with the is_view_of attribute set to source)
    and the first dimension set to a JaggedIntVar. The jagged Tensor resulting
    from this op can then be treated as jagged by other ops aware of the
    jagged Tensor semantics (e.g., elementwise). Importantly, the source
    Tensor is not sufficient for that, as it doesn't carry the necessary
    jagged Tensor metadata (which the jagged Tensor does, in the first
    JaggedIntVar dimension of its shape).

    *Important*: this op is the only right way to create a jagged Tensor.
    The reason is that the offsets Tensors passed to this op get registered
    in the graph and, as a result, can't be optimized out. This wouldn't
    be the case if the jagged Tensor would be "constructed" manually.

    See the docstring of the JaggedIntVar class for more details on the
    jagged Tensor semantics and representation.

    In the backend, the purpose of the make_jagged op is to setup the
    unified offsets representation for the jagged Tensor and to check
    the contents of the rank-1 offsets Tensors for consistency.

    __init__ Args:
        batch_dim : IntVar
            The batch dimension of the jagged Tensor.
            Importantly, this is different from the first dimension of the
            soruce Tensor, as it logically represents the number of variable-
            length sequences encoded by the JaggedIntVar. I.e., the batch_dim
            is B in the sum_B(N_B) representation of the JaggedIntVar.
        jagged_dims : List[JaggedDim]
            The list of jagged dimensions encoded in the JaggedIntVar of the
            resulting jagged Tensor. See the JaggedDim and JaggedIntVar class
            docstrings for the details.

    __call__ Args:
        source : Union[Tensor, List[Tensor]]
            One or more source Tensors of the jagged Tensor(s) created by this op.
            The jagged Tensor is a view of the source Tensor. The main difference
            is that the resulting jagged Tensor's first dimension is set to a
            JaggedIntVar, constructed from the batch_dim, jagged_dims, and the
            offsets_list. The same JaggedIntVar instance is set as the first
            dimension of every resulting jagged Tensor: one for each source
            Tensor in the `source`.
        offsets_list : List[Tensor]
            The list of rank-1 offsets Tensors describing the variable-length
            layout of each of the jagged_dims. There must be exactly as many
            offsets Tensors in the offsets_list as there are JaggedDims in
            the jagged_dims list. Each offsets Tensor is associated with the
            corresponding JaggedDim before constructing a JaggedIntVar from
            them for the resulting jagged Tensor.

    Returns:
        Union[Tensor, List[Tensor]]
            The resulting jagged Tensor or a list thereof, depending on whether
            the `source` argument is a Tensor or a List[Tensor].
    """

    def __init__(
        self,
        batch_dim: IntVar,
        jagged_dims: List[JaggedDim],
        check_sequence_lengths: bool = True,
    ) -> None:
        if not jagged_dims or not all(
            isinstance(dim, JaggedDim) for dim in jagged_dims
        ):
            raise TypeError(
                "jagged_dim must be a non-empty list of JaggedDims, "
                f"but given {jagged_dims}."
            )

        super().__init__()

        self._attrs["op"] = "make_jagged"
        self._attrs["batch_dim"] = batch_dim
        self._attrs["jagged_dims"] = list(jagged_dims)
        self._attrs["check_sequence_lengths"] = check_sequence_lengths

    def _set_jagged_dim_offsets(self, offsets_list: List[Tensor]):
        jagged_dims = self._attrs["jagged_dims"]
        for i, (jagged_dim, offsets) in enumerate(zip(jagged_dims, offsets_list)):
            if jagged_dim.offsets() is not None:
                if jagged_dim.offsets() == offsets:
                    continue
                else:
                    raise ValueError(
                        f"JaggedDim {i} in the jagged_dims already has associated "
                        "offsets != the offsets passed to the make_jagged.__call__."
                    )
            jagged_dim._attrs["offsets"] = offsets

    def __call__(
        self,
        source: Union[Tensor, List[Tensor]],
        offsets_list: List[Tensor],
    ) -> Tensor:
        sources_list = [source] if isinstance(source, Tensor) else source

        if not sources_list:
            raise ValueError("There must be at least one source Tensor in the list.")

        for s in sources_list:
            if len(s._attrs["shape"]) == 0:
                raise ValueError(
                    "The source Tensors must be at least rank-1, but given rank-0."
                )
            if type(s._attrs["shape"][0]) != IntVar:
                raise ValueError(
                    "The source Tensor's first dim (total_length) must be "
                    f"dynamic (IntVar), but given {s._attrs['shape']=}."
                )

        total_length = sources_list[0]._attrs["shape"][0]
        for s in sources_list[1:]:
            if s._attrs["shape"][0] != total_length:
                raise ValueError(
                    "All source Tensors must have the same first (total_length) dimension, "
                    f"but got {s[0]._attrs['shape']=}, {s._attrs['shape']=}."
                )

        if isinstance(total_length, JaggedIntVar):
            # already jagged Tensors
            return source

        jagged_dims = self._attrs["jagged_dims"]
        if len(offsets_list) != len(jagged_dims):
            raise ValueError(
                f"{len(offsets_list)=} must be equal to {len(jagged_dims)=}"
            )
        for offsets in offsets_list:
            if len(offsets._attrs["shape"]) != 1:
                raise ValueError(
                    "The offsets Tensors must be rank-1, "
                    f"but given shape {offsets._attrs['shape']}."
                )
            if offsets._attrs["dtype"] not in ["int32", "int64"]:
                raise TypeError(
                    "The offsets Tensors can be either int32 or int64, "
                    f"but given the Tensor of type {offsets._attrs['dtype']}."
                )

        self._attrs["num_sources"] = len(sources_list)
        self._attrs["inputs"] = sources_list + offsets_list
        self._set_depth()
        self._set_jagged_dim_offsets(offsets_list)

        jagged_int_var = JaggedIntVar(
            batch_dim=self._attrs["batch_dim"],
            jagged_dims=self._attrs["jagged_dims"],
            total_length=total_length,
        )

        outputs = [
            Tensor(
                shape=[jagged_int_var] + s._attrs["shape"][1:],
                src_ops={self},
                is_view_of=s,
            )
            for s in sources_list
        ]
        self._attrs["outputs"] = outputs
        if isinstance(source, Tensor):
            outputs = outputs[0]

        return outputs

    def _get_op_attributes(self):
        return {
            "batch_dim": self._attrs["batch_dim"],
            "jagged_dims": self._attrs["jagged_dims"],
            "check_sequence_lengths": self._attrs["check_sequence_lengths"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def _args_for_pseudo_code(self):
        batch_dim = self._attrs["batch_dim"].pseudo_code()
        jagged_dims = ", ".join(
            [dim.pseudo_code() for dim in self._attrs["jagged_dims"]]
        )
        return [
            f"batch_dim={batch_dim}",
            f"jagged_dims={jagged_dims}",
        ]
