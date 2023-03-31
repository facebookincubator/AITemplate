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
Bind all user-provided constants to the graph.
"""

from typing import Dict, List

from aitemplate.compiler.base import _TorchConstantTensorData, Tensor
from aitemplate.compiler.model import TorchTensor


def bind_constants(graph: List[Tensor], constants: Dict[str, TorchTensor]) -> None:
    """Bind all user-provided constants to the graph. Internally, the constants are
    represented as ConstantTensors. These can be folded, and are packaged into
    the final *.so.

    Parameters
    ----------
    graph : List[Tensor]
        Input graph
    constants : Dict[str, TorchTensor]
        Constants to bind

    """
    if not constants:
        return
    for tensor in graph:
        name = tensor._attrs["name"]
        if name not in constants:
            continue

        if tensor._attrs["data"] is not None:
            raise ValueError(f"Tensor {name} is already bound!")

        if tensor.src_ops():
            raise ValueError(f"Cannot bind non-constant tensor {name}")

        if tensor._attrs["is_input"]:
            raise ValueError(f"Cannot bind input tensor {name}")

        tensor._bind_data(_TorchConstantTensorData(constants[name]))
