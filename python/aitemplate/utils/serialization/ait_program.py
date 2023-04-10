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
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from aitemplate.compiler.base import (
    _HostConstantTensorData,
    _NumpyConstantTensorData,
    _TorchConstantTensorData,
)
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing.test_utils import get_random_torch_tensor


def convert_to_ait_const(const):
    if isinstance(const, bytes):
        return _HostConstantTensorData(const)
    elif isinstance(const, torch.Tensor):
        return _TorchConstantTensorData(const)
    elif isinstance(const, np.ndarray):
        return _NumpyConstantTensorData(const)
    else:
        raise RuntimeError(f"Unknown type ({type(const)}) to convert to AIT Tensor")


class AITBasicProgram:
    def __init__(self):
        """
        Initialize all inputs and constants parameters.
        """
        pass

    def get_constants(self) -> Dict[str, List[int]]:
        """
        Returns a dictionary of the constants.
        The returned dictionary has key as constant name and value as input shape.
        """
        pass

    def get_inputs(self) -> Dict[str, List[IntVar]]:
        """
        Returns a dictionary of the expected inputs.
        The returned dictionary has key as input name and value as input shape.
        """
        pass

    def set_constants(self, constants: Dict[str, Any]):
        """
        Provide a dictionary to set the corresponding constant values.
        The constant value could be bytes/torch.Tensor/numpy.ndarray.
        """
        for k, v in constants.items():
            getattr(self, k)._bind_data(convert_to_ait_const(v))

    def set_default_constants(self, dtype="float16"):
        """
        This function is called to set up default constants
        (ex. constant folded/constants set up by zero padding etc.).
        """
        self.set_all_random_constants(dtype)

    def set_all_random_constants(self, dtype="float16"):
        """
        This function would set all constants into random value.
        """
        const_infos = self.get_constants()
        for k, v in const_infos.items():
            getattr(self, k)._bind_data(
                _TorchConstantTensorData(get_random_torch_tensor(v, dtype))
            )

    def model(self) -> Union[Tensor, Tuple[Tensor]]:
        """
        This function defines the AIT program.
        Returns an output tensor, or a tuple of output tensors.
        """
        pass
