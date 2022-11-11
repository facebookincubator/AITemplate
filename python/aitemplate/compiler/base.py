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
Basic data types of AITemplate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from pprint import pformat
from typing import Any, Dict, List, Set, Union

import numpy as np

from aitemplate.compiler.dtype import get_dtype_size, normalize_dtype

from aitemplate.compiler.stable_set import StableSet
from aitemplate.utils.torch_utils import torch_dtype_to_string

from ..utils.tensor_utils import wrap_dim
from .op_registry import OP_REGISTRY

# pylint: disable=C0206,W0613,C0201,W0102,W0231,W0233


class Node(ABC):
    """Base class of Tensor, Operator, etc."""

    def __init__(self) -> None:
        """
        Initializes self._attrs field, which is a dict that stores
        all attributes for this Node.
        Basic attributes include:
            * name: str, name of the node.
            * depth: int, depth of the node in a graph. None if this is not applicable.
            * nop: bool, marks whether this node is a no-operation.
        Child classes add their own attributes to this dict.
        """
        super().__init__()
        self._attrs: Dict[str, Any] = {"name": None, "depth": 0, "nop": False}

    def __str__(self) -> str:
        """Returns a string version of this object."""
        return pformat(self._attrs, indent=2, depth=2)

    def __repr__(self) -> str:
        """Returns a string containing a printable representation of this object."""
        return self.__str__()

    @abstractmethod
    def pseudo_code(self, with_shape: bool = False) -> str:
        """Returns a string containing pseudo code of this object.

        Parameters
        ----------
        with_shape: bool
            Marks whether to include shape info in the returned pseudo code.

        Returns
        ----------
        str
            Pseudo code.
        """
        pass


class IntVar(Node):
    """
    An IntVar represents a dynamic dimension.
    IntVar and IntImm (see below) are used together to represent a Tensor's shape.
    """

    def __init__(
        self,
        values: List[int],
        name: str = None,
    ) -> None:
        """Initializes an IntVar.

        Parameters
        ----------
        values : list[int]
            A list of possible values of this dynamic dimension.
            len(values) must be >= 2.

            When len(values) == 2, the values are treated as a lower bound and an upper bound.
            Both upper bound and lower bound are inclusive.
            This is the default use case.

            When len(values) > 2, the first / last values are treated as lower / upper bounds,
            and the other values are used for internal profiling purpose.
            This is a legacy use case.

        name : str, optional
            Name of this dimension, by default None.
            This field must be set for dims which are used by input tensors.
        """
        super().__init__()
        self._attrs["name"] = name

        if values is None or len(values) < 2:
            raise RuntimeError(
                "IntVar 'values' field must have at least 2 values! values: {}, name: {}".format(
                    values, name
                )
            )
        if min(values) < 0:
            raise RuntimeError(
                "IntVar has < 0 value! values: {}, name: {}".format(values, name)
            )
        self._attrs["values"] = sorted(set(values))
        if len(self._attrs["values"]) == 1:
            self._attrs["values"] = self._attrs["values"] * 2

    def __str__(self) -> str:
        return pformat(self._attrs, indent=2)

    def __eq__(self, another: Any) -> bool:
        return (
            isinstance(another, IntVar)
            and self._attrs["values"] == another._attrs["values"]
            and self._attrs["name"] == another._attrs["name"]
        )

    def __hash__(self) -> int:
        return hash((self._attrs["name"], tuple(self._attrs["values"])))

    def lower_bound(self) -> int:
        """Returns lower bound of this dynamic dim."""
        return self._attrs["values"][0]

    def upper_bound(self) -> int:
        """Returns upper bound of this dynamic dim."""
        return self._attrs["values"][-1]

    def pseudo_code(self, with_shape=False) -> str:
        return (
            self._attrs["name"]
            if self._attrs["name"] is not None
            else f"IntVar({str(self._attrs['values'])})"
        )


class IntImm(IntVar):
    """
    An IntImm represents a static dimension.
    IntVar (see above) and IntImm are used together to represent a Tensor's shape.
    """

    def __init__(
        self,
        value: int,
        name: str = None,
    ) -> None:
        """Initializes an IntImm.

        Parameters
        ----------
        value : int
            Value of this static dimension.

        name : str, optional
            Name of this dimension, by default None.
            This field must be set for dims which are used by input tensors.
        """

        if not isinstance(value, int):
            raise RuntimeError(
                "IntImm only takes an int value! Name: {}, current value: {}".format(
                    name, value
                )
            )

        Node.__init__(self)  # pylint: disable=W0233
        self._attrs["name"] = name
        self._attrs["values"] = [value]

    def __eq__(self, another: Union[int, IntVar]) -> bool:
        if isinstance(another, int):
            return self.value() == another

        return (
            isinstance(another, IntImm)
            and self._attrs["values"] == another._attrs["values"]
        )

    def value(self) -> int:
        """Returns value of this IntImm."""
        return self._attrs["values"][0]

    def pseudo_code(self, with_shape=False) -> str:
        return str(self.value())


def get_aligned_size(shape: List[IntVar], dtype: str, alignment: int = 64) -> int:
    """Returns aligned size (in bytes) of given shape and dtype.

    Parameters
    ----------
    shape: List[IntVar]
        A list of IntVars, which represents the shape of a Tensor.
    dtype: str
        A data type string.
    alignment: int
        Alignment requirement (in bytes). Default alignment is 64 bytes.

    Returns
    ----------
    int
        Size (in bytes) of this shape with dtype, aligned in alignment bytes.
    """

    size = reduce(lambda cur, dim: cur * dim.upper_bound(), shape, 1)
    size = size * get_dtype_size(dtype)
    if size % alignment != 0:
        size = int((size // alignment + 1) * alignment)
    return size


class _ConstantTensorData(ABC):
    """
    Represents data to be stored in a Tensor.
    The data can be owned or unowned; each subclass should
    implement its own setup and cleanup logic.

    Note that this class is different from the blobs that are used
    in the Python API (e.g. in Run(), compile_model). Those
    blobs must represent unowned GPU memory; subclasses of _ConstantTensorData
    may be owned or unowned, and may or may not reside in host memory.

    Why is this separate class useful? During compilation, we have no way to
    allocate memory on the GPU, so host memory must be used to store tensors that
    we introduce (e.g. padding tensors). At the same time, when lowering PyTorch models,
    we may want to store owned GPU data in the graph.
    """

    def __init__(self, dtype: str):
        super().__init__()
        self.dtype = normalize_dtype(dtype)

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Converts the stored data to a byte string.
        Called during codegen to save the ConstantTensor to the
        .so.
        """
        pass

    def size(self) -> int:
        """
        The number of bytes stored. Should be equal to
        len(self.to_bytes()).
        """
        return len(self.to_bytes())

    def is_dtype(self, dtype: str) -> bool:
        return normalize_dtype(dtype) == self.dtype

    def __len__(self) -> int:
        return self.size()


class _HostConstantTensorData(_ConstantTensorData):
    """
    The simplest possible _ConstantTensorData; just a
    lightweight wrapper around some host data.
    """

    def __init__(self, data: bytes, dtype: str = "float16"):
        super().__init__(dtype)
        self.data = data

    def to_bytes(self) -> bytes:
        return self.data


class _TorchConstantTensorData(_ConstantTensorData):
    """
    Wraps a torch.Tensor for storage in _ConstantTensorData.
    """

    def __init__(self, tensor):
        super().__init__(torch_dtype_to_string(tensor.dtype))
        self.tensor = tensor

    def to_bytes(self) -> bytes:
        return self.tensor.cpu().detach().numpy().tobytes()

    def size(self) -> int:
        """
        Override size() to avoid D2H copy.
        """
        return self.tensor.element_size() * self.tensor.nelement()


class _NumpyConstantTensorData(_ConstantTensorData):
    """
    Wraps an ndarray for storage in _ConstantTensorData.
    """

    def __init__(self, arr: np.ndarray):
        super().__init__(str(arr.dtype))
        self.arr = arr

    def to_bytes(self) -> bytes:
        return self.arr.tobytes()


class Tensor(Node):
    """
    A Tensor represents a piece of data, which is used as an input / output of an Operator.
    Both Tensor and Operator are used at model compilation stage.
    """

    def __init__(
        self,
        shape: List[IntVar],
        name: str = None,
        src_ops: StableSet[Node] = None,
        dst_ops: StableSet[Node] = None,
        dtype: str = "float16",
        is_input: bool = False,
        is_output: bool = False,
        value: Any = None,
        is_view_of: Any = None,
        check_nan_and_inf: bool = False,
        check_outputs: bool = False,
    ) -> None:
        """Initializes a Tensor.

        Parameters
        ----------
        shape : List[IntVar]
            Shape of this Tensor.
        name : str, optional
            Name of this Tensor. By default it's None.
        src_ops : Set[Node], optional
            Source operators of this Tensor which write to this Tensor.
            By default it's an empty set.
        dst_ops : Set[Node], optional
            Destination operators of this Tensor which take this Tensor as
            one of their inputs.
            By default it's an empty set.
        dtype : str, optional
            Date type of this Tensor. By default it's "float16".
        is_input : bool, optional
            Whether this Tensor is an input Tensor of a graph.
            Note that constant Tensors (e.g. weights) are NOT input Tensors.
        is_output : bool, optional
            Whether this Tensor is an output Tensor of a graph.
        value : Any, optional
            The value of this Tensor. When value is set and shape is an
            empty list, this Tensor is used to represent a number.
        is_view_of : Any, optional
            Whether this Tensor is a view of another Tensor.
        check_nan_and_inf : bool, optional
            Whether or not to check this tensor is nan or inf during runtime.
        check_outputs : bool, optional
            Whether or not to print this tensor's value out during runtime.
        """
        super().__init__()
        self._attrs["shape"] = self._convert_shape(shape)
        self._attrs["name"] = name
        self._attrs["src_ops"] = (
            StableSet(src_ops) if src_ops is not None else StableSet()
        )
        self._attrs["dst_ops"] = (
            StableSet(dst_ops) if dst_ops is not None else StableSet()
        )
        self._attrs["dtype"] = dtype
        self._attrs["is_output"] = is_output
        self._attrs["is_input"] = is_input
        self._attrs["is_param"] = False

        # True if this is an internal tensor that aliases an output through
        # a view. Set up in mark_param_tensor
        self._attrs["has_output_aliases"] = False

        # For special views. When an output is a view of an input/constant/other
        # output, this attribute points to that view. Note that this is not the
        # same as is_view_of if the output is a view of a view. This is set up
        # in the mark_param_tensor graph pass.
        self._attrs["external_tensor"] = None

        # link to original tensor if this tensor is a view
        self._attrs["is_view_of"] = is_view_of

        if is_view_of:
            self._attrs["dtype"] = is_view_of._attrs["dtype"]

        self._attrs["value"] = value
        src_deps = [src_op._attrs["depth"] for src_op in self._attrs["src_ops"]]
        self._attrs["depth"] = max(src_deps) + 1 if len(src_deps) > 0 else 0

        # Offset into internal memory slab, set by memory planning
        self._attrs["offset"] = None

        # Data to be bound for constant folding. See _bind_data.
        self._attrs["data"] = None

        self._attrs["check_nan_and_inf"] = check_nan_and_inf
        self._attrs["check_outputs"] = check_outputs

    def __str__(self) -> str:
        output = {}
        for key in self._attrs.keys():
            if key in ("src_ops", "dst_ops") and self._attrs[key] is not None:
                output[key] = [x._attrs["name"] for x in self._attrs[key]]
            else:
                output[key] = self._attrs[key]
        return pformat(output, indent=2)

    def _convert_shape(self, shape: List[Union[int, IntVar]]) -> List[IntVar]:
        """
        Converts from a list of ints / IntVars to a list of IntVars.
        """
        ret = []
        for v in shape:
            if isinstance(v, int):
                ret.append(IntImm(v))
            elif isinstance(v, IntVar):
                ret.append(v)
            else:
                raise RuntimeError(f"Unsupported dim type: {type(v)}, dim: {v}")
        return ret

    def shape(self) -> List[IntVar]:
        """
        Returns the shape of the tensor.
        It should not be used directly in IR.
        """
        return self._attrs["shape"]

    def _rank(self) -> int:
        """
        Returns the rank of the tensor.
        It should not be used directly in IR.
        """
        return len(self._attrs["shape"])

    def _size(self, dim) -> IntVar:
        """
        Gets the size of tensor at dim=dim.
        dim must be between [-rank, rank - 1].
        It should not be used directly in IR, use ops.size(dim) instead.
        """
        return self._attrs["shape"][wrap_dim(dim, self._rank())]

    def dtype(self) -> str:
        """Returns Tensor's data type str."""
        return self._attrs["dtype"]

    def src_ops(self) -> Set[Operator]:
        """Returns a set of source operators which write to this Tensor."""
        return self._attrs["src_ops"]

    def dst_ops(self) -> Set[Operator]:
        """Returns a set of destination operators which read from this Tensor."""
        return self._attrs["dst_ops"]

    def is_a_const_num(self) -> bool:
        """Returns whether this Tensor represents a constant number."""
        return len(self._attrs["shape"]) == 0 and self._attrs["value"] is not None

    def size_bytes(self, alignment: int = 1) -> int:
        """Returns acutal size (in bytes) of this Tensor."""
        return get_aligned_size(self._attrs["shape"], self.dtype(), alignment)

    def pseudo_code(self, with_shape=True) -> str:
        name = self._attrs["name"]
        if name is None:
            name = "None"

        args = [f"name={name}"]

        if with_shape:
            shapes = ", ".join([dim.pseudo_code() for dim in self._attrs["shape"]])
            args.append(f"shape=[{shapes}]")

        data = self._attrs["data"]
        if data is not None:
            args.append(f"data=({data.size()} bytes)")

        return f"Tensor({', '.join(args)})"

    def _bind_data(self, data: _ConstantTensorData) -> None:
        """
        Bind some data to this tensor.
        - This tensor must not have any src_ops().
        - The provided data's size in bytes much match the maximum size of this tensor

        Tensors with bound data can participate in constant folding.
        """
        if self.src_ops():
            raise ValueError(
                f"Cannot bind tensor {self._attrs['name']}; {len(self.src_ops())=} > 0"
            )
        dtype = self._attrs["dtype"]
        if not data.is_dtype(dtype):
            raise ValueError(
                f"data's dtype did not match: expected {dtype}, got {data.dtype}"
            )
        tensor_size = self.size_bytes(alignment=1)
        if tensor_size != len(data):
            raise ValueError(
                (
                    "ConstantTensor's maximum size is not equal to len(data)! "
                    f"Got {len(data)=}, but expected at least {tensor_size} bytes. "
                    "Check that the ConstantTensor's size and dtype are correct."
                )
            )
        self._attrs["data"] = data

    def __add__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("ADD")(self, other)

    def __radd__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("ADD")(other, self)

    def __sub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("SUB")(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("SUB")(other, self)

    def __mul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("MUL")(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("MUL")(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("DIV")(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("DIV")(other, self)

    def __neg__(self) -> Tensor:
        return OP_REGISTRY.get("MUL")(-1, self)


def _create_host_zero_tensor(
    shape: List[Union[int, IntVar]],
    name: str = None,
    dst_ops: Set[Node] = None,
    dtype: str = "float16",
    is_output: bool = False,
):
    """
    Create a zero tensor stored on the host machine.
    """
    shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in shape]
    zeros = _HostConstantTensorData(
        b"\x00" * get_aligned_size(shape, dtype, alignment=1), dtype=dtype
    )
    tensor = Tensor(shape, name, dst_ops=dst_ops, dtype=dtype, is_output=is_output)
    tensor._bind_data(zeros)
    return tensor


class IntVarTensor(Tensor):
    """
    A special tensor which represents an IntImm / IntVar.
    This Tensor can be used as inputs of some Operators (e.g. reshape, layernorm).
    An IntVarTensor instead of IntVar is used here to keep reference to
    src_ops and dst_ops.
    """

    def __init__(
        self,
        int_var: IntVar,
        name: str = None,
        src_ops: Set[Node] = None,
        dst_ops: Set[Node] = None,
        dtype: str = "float16",
        is_input: bool = False,
        is_output: bool = False,
        value: Any = None,
        is_view_of: Any = None,
    ) -> None:
        """Initializes an IntVar Tensor.

        Parameters
        ----------
        int_var: IntVar
            The underlying IntVar variable.
        """
        shape = []
        super().__init__(
            shape,
            name,
            src_ops,
            dst_ops,
            dtype=dtype,
            is_input=is_input,
            is_output=is_output,
        )
        self._attrs["int_var"] = int_var

    def pseudo_code(self, with_shape=True) -> str:
        return f"IntVarTensor({self._attrs['int_var'].pseudo_code()})"

    def __add__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_ADD")(self, other)

    def __radd__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_ADD")(other, self)

    def __sub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_SUB")(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_SUB")(other, self)

    def __mul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_MUL")(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_MUL")(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_DIV")(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        return OP_REGISTRY.get("INT_DIV")(other, self)


class DynamicProfileStrategy(Enum):
    """Dynamic profiling stategy enum.
    Instances are used to select profiling strategy when there are dynamic dims.
    """

    # Always use an IntVar's min value to profile.
    MIN = 1
    # Always use an IntVar's max value to profile.
    MAX = 2
    # Profiling according to an IntVar's value list.
    # For testing purpose only.
    HINTS = 3


@dataclass
class ExecItem:
    """A data class to store profiling info."""

    profiling_key: str
    exec_cond: str
    algo: str


class Operator(Node):
    """Base class for all operators"""

    def __init__(self) -> None:
        """Initializes the operator."""
        super().__init__()
        self._attrs["inputs"] = None
        self._attrs["has_profiler"] = False

    def __call__(self, *args: List[Tensor]) -> List[Tensor]:
        """Performs offline shape inference and constructs the model graph.

        Parameters
        -------
        *args : List[Tensor]
            Input tensors.

        Returns
        -------
        List[Tensor]
            Output tensors.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def _set_depth(self) -> None:
        """
        Sets operator depth and dst_ops.
        This function must be called by each operator subclass inside
        __call__() method once self._attrs["inputs"] is set.
        """
        max_depth = 0
        if self._attrs["inputs"] is not None:
            for inp in self._attrs["inputs"]:
                max_depth = max(max_depth, inp._attrs["depth"])
                inp._attrs["dst_ops"].add(self)
        self._attrs["depth"] = max_depth

    def __str__(self) -> str:
        """Generates a debug string."""
        output = {}
        for key in self._attrs.keys():
            if (
                key in ("inputs", "args", "outputs", "original_inputs")
                and self._attrs[key] is not None
            ):
                output[key] = [x._attrs["name"] for x in self._attrs[key]]
            else:
                output[key] = self._attrs[key]
        return pformat(output, indent=2)

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """Generates source files for profiling purpose.

        Parameters
        ----------
        workdir : str, optional
            The directory to generate source files.
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
        return

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=DynamicProfileStrategy.MAX,
    ) -> None:
        """Selects the fastest kernel configurations.

        Parameters
        ----------
        workdir : str, optional
            The directory which contains source files, by default "./"
        devices: list, optional
            A list of device ids which can be used for profiling.
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            Profiling strategy used when there are dynamic dims.
            By default MAX is used, i.e. to profile a dynamic range, an upper bound will be used.
        """

        return

    def gen_function(self) -> str:
        """Generates function source code string.

        Returns
        -------
        str : a string which contains C++ function implementation source code.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("gen_function is not defined for {}".format(self))

    # APIs below are for graph transformations.
    def replace_input_tensor(self, old_tensor, new_tensor) -> None:
        """Replaces old_tensors in self._attrs["inputs"] with new_tensor.

        Parameters
        ----------
        old_tensor: Tensor
            The old tensor to be replaced.
        new_tensor: Tensor
            The new tensor.

        Returns
        -------
        None.
        """

        self._attrs["inputs"] = [
            new_tensor if tensor is old_tensor else tensor
            for tensor in self._attrs["inputs"]
        ]

    def _get_op_attributes(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the core attributes of the op.
        The core attributes are attributes that are required to create an op, for
        example, the FuncEnum for a elementwise op.

        This is used when we need to copy the op with identical behaviour.

        Parameters
        ----------
        None

        Returns
        -------
        Dict of attributes
        """

        return {}

    # APIs below are for pseudo code generation.
    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"]

    def _outputs_for_pseudo_code(self):
        return self._attrs["outputs"]

    def _args_for_pseudo_code(self):
        return []

    def _pseudo_code_helper(self, node: Any, with_shape: bool) -> str:
        if isinstance(node, list):
            if len(node) > 3 and isinstance(node[0], Tensor):
                return ",\n".join(self._pseudo_code_helper(n, with_shape) for n in node)
            else:
                return ", ".join(self._pseudo_code_helper(n, with_shape) for n in node)
        if isinstance(node, Node):
            return node.pseudo_code(with_shape)
        return str(node)

    def pseudo_code(self, with_shape=True):
        args = self._pseudo_code_helper(self._args_for_pseudo_code(), with_shape)
        inputs = self._pseudo_code_helper(self._inputs_for_pseudo_code(), with_shape)
        outputs = self._pseudo_code_helper(self._outputs_for_pseudo_code(), with_shape)
        return f"({outputs}) \n= {self._attrs['op']}({args})(\n{inputs})\n"
