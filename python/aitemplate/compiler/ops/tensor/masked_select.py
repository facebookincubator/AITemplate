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
Define masked_select op
"""
import logging
from typing import List

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntVar, Operator, Tensor

from aitemplate.compiler.dtype import get_dtype_size

from aitemplate.utils import shape_utils


_LOGGER = logging.getLogger(__name__)


class masked_select(Operator):
    """
    Returns a 1D tensor containing elements of the input tensor selected by the boolean mask,
    similar to `torch.masked_select`.

    Args:
        input (Tensor): the source tensor.
        mask (Tensor, boolean): the shapes of the mask tensor and the input tensor do not need
            to match, but they must be broadcastable.

    Returns:
        output: 1D tensor of length equal to the total number of elements in broadcast shape
            deduced from input and mask. The result is contained in the first `num_nonmasked`
            elements of output. The rest of the output tensor is not meaningful.
        num_nonmasked: number of the non-masked elements from the input, i.e. the length of the
            significant part of output.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "masked_select"
        self._attrs["workspace"] = 0
        self._attrs["max_shape"] = None

    def _infer_shape(self, x: Tensor, mask: Tensor) -> List[IntVar]:
        input_shape = x._attrs["shape"]
        mask_shape = mask._attrs["shape"]
        broadcastable, max_shape = shape_utils.get_broadcast_max_shape(
            input_shape, mask_shape
        )
        if not broadcastable:
            raise RuntimeError(
                "Tensor shapes of input and mask are not broadcastable! Shape1: {}, shape2: {}".format(
                    input_shape, mask_shape
                )
            )
        self._attrs["max_shape"] = max_shape
        numel = 1
        for dim in max_shape:
            numel *= dim.upper_bound()

        # Allocate temporary buffer. This empirical formula for size is deduced by looking at necessary
        # memory to expand input/mask and the buffer sizes requested by cub::DeviceSelect::Flagged for
        # different input sizes.
        input_workspace_size = (input_shape != max_shape) * get_dtype_size(
            x._attrs["dtype"]
        )
        mask_workspace_size = (mask_shape != max_shape) * 1  # bool
        self._attrs["workspace"] = (
            numel * (input_workspace_size + mask_workspace_size) + numel // 128 + 1024
        )
        _LOGGER.debug(
            f'Allocating {self._attrs["workspace"]} bytes for temporary buffer of masked_select op'
        )

        # Output size can range from 0 (when all mask elements are False) to the total number of
        # elements in the input (when all mask elements are True).
        return [IntVar(values=(0, numel))]

    def __call__(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> List[Tensor]:
        dtype = mask._attrs["dtype"]
        if dtype != "bool":
            raise RuntimeError("Expected mask of dtype bool, but got {}".format(dtype))
        self._attrs["inputs"] = [x, mask]
        self._set_depth()
        output_shape = self._infer_shape(x, mask)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])

        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
