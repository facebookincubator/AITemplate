# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from pprint import pformat
from typing import Any, List, Set, Union

import jinja2

import numpy as np

from aitemplate.utils.torch_utils import torch_dtype_to_string

from ..utils.tensor_utils import wrap_dim

# pylint: disable=C0206,W0613,C0201,W0102,W0231,W0233


_DTYPE2BYTE = {
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
}


def get_dtype_size(dtype: str) -> int:
    if dtype not in _DTYPE2BYTE:
        raise KeyError(f"Unknown dtype: {dtype}. Expected one of {_DTYPE2BYTE.keys()}")
    return _DTYPE2BYTE[dtype]


class Node(ABC):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self) -> None:
        """[summary]"""
        super().__init__()
        self._attrs = {"name": None, "depth": 0, "nop": False}

    def __str__(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        return pformat(self._attrs, indent=2, depth=2)

    def __repr__(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        return self.__str__()

    @abstractmethod
    def pseudo_code(self, with_shape=False):
        pass


class IntVar(Node):
    """An IntVar represents a dynamic dimension."""

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

    def __hash__(self):
        return hash((self._attrs["name"], tuple(self._attrs["values"])))

    def lower_bound(self):
        return self._attrs["values"][0]

    def upper_bound(self):
        return self._attrs["values"][-1]

    def pseudo_code(self, with_shape=False):
        return (
            self._attrs["name"]
            if self._attrs["name"] is not None
            else f"IntVar({str(self._attrs['values'])})"
        )


class IntImm(IntVar):
    """An IntImm represents a static dimension."""

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
        return self._attrs["values"][0]

    def pseudo_code(self, with_shape=False):
        return str(self.value())


def get_aligned_size(shape: List[IntVar], dtype: str, alignment: int = 64):
    size = reduce(lambda cur, dim: cur * dim.upper_bound(), shape, 1)
    size = size * get_dtype_size(dtype)
    if size % alignment != 0:
        size = int((size // alignment + 1) * alignment)
    return size


class Tensor(Node):
    """[summary]

    Parameters
    ----------
    Node : [type]
        [description]
    """

    def __init__(
        self,
        shape: List[IntVar],
        name: str = None,
        src_ops: Set[Node] = None,
        dst_ops: Set[Node] = None,
        dtype: str = "float16",
        is_input: bool = False,
        is_output: bool = False,
        value: Any = None,
        is_view_of: Any = None,
    ) -> None:
        """[summary]

        Parameters
        ----------
        shape : List[IntVar]
            [description]
        name : str, optional
            [description], by default an empty set().
        src_ops : Set[Node], optional
            [description], by default an empty set()
        dst_ops : Set[Node], optional
            [description], by default None
        dtype : str, optional
            [description], by default "float16"
        """
        super().__init__()
        self._attrs["shape"] = self._convert_shape(shape)
        self._attrs["name"] = name
        self._attrs["src_ops"] = src_ops if src_ops is not None else set()
        self._attrs["dst_ops"] = dst_ops if dst_ops is not None else set()
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

        self._attrs["value"] = value
        src_deps = [src_op._attrs["depth"] for src_op in self._attrs["src_ops"]]
        self._attrs["depth"] = max(src_deps) + 1 if len(src_deps) > 0 else 0

    def __str__(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        output = {}
        for key in self._attrs.keys():
            if key in ("src_ops", "dst_ops") and self._attrs[key] is not None:
                output[key] = [x._attrs["name"] for x in self._attrs[key]]
            else:
                output[key] = self._attrs[key]
        return pformat(output, indent=2)

    def print_nan_ratio(self) -> str:
        print_nan_ratio_template = jinja2.Template(
            """
{{indent}}{
{{indent}}  int64_t nan_num = 0, inf_num = 0;
{{indent}}  int64_t elem_cnt = {{elem_cnt}};
{{indent}}  auto values = reinterpret_cast<half *>({{name}});
{{indent}}  for (int64_t i = 0; i < elem_cnt; i++) {
{{indent}}    float v = (float)(*(values + i));
{{indent}}    if (isnan(v)) {
{{indent}}      nan_num += 1;
{{indent}}    } else if (isinf(v)) {
{{indent}}      inf_num += 1;
{{indent}}    }
{{indent}}  }
{{indent}}  if (nan_num > 0 || inf_num > 0){
{{indent}}    LOG(INFO) << __FILE__ << ":" << __LINE__ << ": Tensor {{name}} has " << nan_num/elem_cnt*100 << "% NaNs, " << inf_num/elem_cnt*100 << "% Infs.";
{{indent}}  }
{{indent}}}
"""
        )
        elem_cnt = "*".join([idx.pseudo_code() for idx in self._attrs["shape"]])
        if not elem_cnt:
            return ""

        return print_nan_ratio_template.render(
            elem_cnt=elem_cnt, indent="    ", name=self._attrs["name"]
        )

    def _convert_shape(self, shape):
        """[summary]

        Parameters
        ----------
        shape : [type]
            [description]

        Returns
        -------
        [type]
            [description]
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

    def shape(self):
        """Return the shape of the tensor.
        It should not be used directly in IR.
        """
        return self._attrs["shape"]

    def _rank(self):
        """Return the rank of the tensor.
        It should not be used directly in IR.
        """
        return len(self._attrs["shape"])

    def _size(self, dim):
        """Get the size of tensor at dim=dim.
        dim must be between [-rank, rank - 1].
        It should not be used directly in IR, use ops.size(dim) instead.
        """
        return self._attrs["shape"][wrap_dim(dim, self._rank())]

    def dtype(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._attrs["dtype"]

    def src_ops(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._attrs["src_ops"]

    def dst_ops(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._attrs["dst_ops"]

    def is_a_const_num(self):
        return len(self._attrs["shape"]) == 0 and self._attrs["value"] is not None

    def size_bytes(self, alignment: int = 1):
        return get_aligned_size(self._attrs["shape"], self.dtype(), alignment)

    def pseudo_code(self, with_shape=True):
        name = self._attrs["name"]
        if name is None:
            name = "None"
        if not with_shape:
            return f"Tensor(name={name})"
        shapes = ", ".join([dim.pseudo_code() for dim in self._attrs["shape"]])
        return f"Tensor(name={name}, shape=[{shapes}])"


class IntVarTensor(Tensor):
    """A special tensor which represents an IntImm / IntVar."""

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
        """[summary]

        Parameters
        ----------
        name : str, optional
            [description], by default an empty set().
        src_ops : Set[Node], optional
            [description], by default an empty set()
        dst_ops : Set[Node], optional
            [description], by default None
        dtype : str, optional
            [description], by default "float16"
        """
        shape = []
        super().__init__(
            shape,
            name,
            src_ops,
            dst_ops,
            is_input=is_input,
            is_output=is_output,
        )
        self._attrs["int_var"] = int_var

    def pseudo_code(self, with_shape=True):
        return f"IntVarTensor({self._attrs['int_var'].pseudo_code()})"


class ConstantTensorData(ABC):
    """
    Represents data to be stored in a ConstantTensor.
    The data can be owned or unowned; each subclass should
    implement its own setup and cleanup logic.
    """

    def __init__(self, dtype: str):
        super().__init__()
        self.dtype = self._normalize_dtype(dtype)

    def _normalize_dtype(self, dtype: str) -> str:
        if dtype == "int":
            return "int32"
        if dtype == "float":
            return "float32"
        return dtype

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
        return self._normalize_dtype(dtype) == self.dtype

    def __len__(self) -> int:
        return self.size()


class HostConstantTensorData(ConstantTensorData):
    """
    The simplest possible ConstantTensorData; just a
    lightweight wrapper around some host data.
    """

    def __init__(self, data: bytes, dtype: str = "float16"):
        super().__init__(dtype)
        self.data = data

    def to_bytes(self) -> bytes:
        return self.data


class TorchConstantTensorData(ConstantTensorData):
    """
    Wraps a torch.Tensor for storage in ConstantTensorData.
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


class NumpyConstantTensorData(ConstantTensorData):
    """
    Wraps an ndarray for storage in ConstantTensorData.
    """

    def __init__(self, arr: np.ndarray):
        super().__init__(str(arr.dtype))
        self.arr = arr

    def to_bytes(self) -> bytes:
        return self.arr.tobytes()


class ConstantTensor(Tensor):
    """
    A special Tensor that represents a constant which is
    known when the graph is constructed. The constructor takes
    data stored in some subclass of ConstantTensorData. The bytes
    (obtained via data.to_bytes()) are stored in the .so at compilation time.

    One use case for ConstantTensor is introducing
    constants during graph passes. For example, many ops will want
    to pad their inputs with zeros. This can be done with a graph
    pass that introduces a new op: concatenate(input, zeros). From the
    user's perspective, zeros is an "invisible" constant; it's not in their
    model definition, it's introduced by us. It is therefore unreasonable to
    expect the user to provide the correct value for 'zeros' at compilation time.
    The solution in this case is to define zeros as a ConstantTensor.

    Note that constant folding is only applied to ConstantTensors. This is required;
    we can't fold constants if we don't know their values!

    Note that there are strict restrictions on the shape argument for a ConstantTensor:
    the maximum size of the tensor must match len(data).

    Note that dynamic shapes are technically allowed here -- it's useful for e.g.
    padding tensors. But most use cases will want to use static shapes, using
    dynamic shapes can lead to unexpected results.
    """

    def __init__(
        self,
        data: ConstantTensorData,
        shape: List[IntVar],
        name: str = None,
        dst_ops: Set[Node] = None,
        dtype: str = "float16",
        is_output: bool = False,
    ):
        super().__init__(
            shape, name=name, dst_ops=dst_ops, dtype=dtype, is_output=is_output
        )
        if not data.is_dtype(dtype):
            raise ValueError(
                f"ConstantTensor's dtype did not match: expected {dtype}, got {data.dtype}"
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

    @staticmethod
    def create_host_zero_tensor(
        shape: List[IntVar],
        name: str = None,
        dst_ops: Set[Node] = None,
        dtype: str = "float16",
        is_output: bool = False,
    ):
        """
        Create a zero tensor stored on the host machine.
        """
        zeros = HostConstantTensorData(
            b"\x00" * get_aligned_size(shape, dtype, alignment=1)
        )
        return ConstantTensor(zeros, shape, name, dst_ops, dtype, is_output)

    def pseudo_code(self, with_shape=False):
        return f"Constant{super().pseudo_code(with_shape)}"


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
        List[Tensor] : Output tensors.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def _set_depth(self) -> None:
        """Sets operator depth and dst_ops.
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

    def gen_function_decl(self) -> str:
        """Generates function declaration string.

        Returns
        -------
        str : a string which contains C++ function declaration.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "gen_function_decl is not defined for {}".format(self)
        )

    def gen_function_call(self) -> str:
        """Generates function call string.

        Returns
        -------
        str : a string which contains C++ function call statements.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "gen_function_call is not defined for {}".format(self)
        )

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

    # APIs for graph transformations.
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

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"]

    def _outputs_for_pseudo_code(self):
        return self._attrs["outputs"]

    def _args_for_pseudo_code(self):
        return []

    def _pseudo_code_helper(self, node: Any, with_shape: bool) -> str:
        if isinstance(node, list):
            return ", ".join(self._pseudo_code_helper(n, with_shape) for n in node)
        if isinstance(node, Node):
            return node.pseudo_code(with_shape)
        return str(node)

    def pseudo_code(self, with_shape=True):
        args = self._pseudo_code_helper(self._args_for_pseudo_code(), with_shape)
        inputs = self._pseudo_code_helper(self._inputs_for_pseudo_code(), with_shape)
        outputs = self._pseudo_code_helper(self._outputs_for_pseudo_code(), with_shape)
        return f"({outputs}) = {self._attrs['op']}({args})({inputs})"
