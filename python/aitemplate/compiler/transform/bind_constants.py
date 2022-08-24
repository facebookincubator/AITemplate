# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Dict, List

from aitemplate.compiler.base import ConstantTensor, Tensor, TorchConstantTensorData
from aitemplate.compiler.transform.transform_utils import replace_tensor
from aitemplate.testing.model import TorchTensor


def bind_constants(graph: List[Tensor], constants: Dict[str, TorchTensor]) -> None:
    """
    Bind all user-provided constants to the graph. Internally, the constants are
    represented as ConstantTensors. These can be folded, and are packaged into
    the final *.so.
    """
    for idx, tensor in enumerate(graph):
        name = tensor._attrs["name"]
        if name not in constants:
            continue

        if isinstance(tensor, ConstantTensor):
            raise ValueError(f"Tensor {name} is already bound!")

        if tensor.src_ops():
            raise ValueError(f"Cannot bind non-constant tensor {name}")

        if tensor._attrs["is_input"]:
            raise ValueError(f"Cannot bind input tensor {name}")

        data = TorchConstantTensorData(constants[name])
        new_tensor = ConstantTensor(
            data,
            tensor._attrs["shape"],
            name=tensor._attrs["name"],
            dtype=tensor._attrs["dtype"],
            is_output=tensor._attrs["is_output"],
        )
        replace_tensor(tensor, new_tensor)
        graph[idx] = new_tensor
