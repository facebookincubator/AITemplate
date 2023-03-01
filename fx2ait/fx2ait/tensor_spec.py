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
import logging
from typing import Any, List

import torch
from aitemplate.compiler.public import IntImm, IntVar

logger: logging.Logger = logging.getLogger(__name__)


class TensorSpec:
    def __init__(self, shape: List[IntVar], dtype: torch.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorSpec):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self.shape) != len(other.shape):
            return False
        for d1, d2 in zip(self.shape, other.shape):
            if d1 != d2:
                return False
        return True

    def __str__(self) -> str:
        return "TensorSpec[shape=[{}],dtype={}]".format(
            ",".join([str(d) for d in self.shape]), self.dtype
        )

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_two_input_lists(
        cls, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]
    ) -> List["TensorSpec"]:
        """
        This function is useful when we expect multiple dynamic dims.

        The parent graph can receive two sets of inputs:
        1. with min dynamic dim values,
        2. with max dynamic dim values.

        After FX splitter logic is applied and lowerable subgraph sample inputs
        are inferred, we make two assumptions:
        1. two lists of inferred inputs will differ at dynamic dimensions,
        2. the difference numbers will be the dynamic ranges, i.e. min and max.

        TODO: The assumptions above are not ideal, and, in theory, we should do
        symbolic shape propagation using something like SymPy.
        """
        if len(inputs1) != len(inputs2):
            raise ValueError(
                f"Different number of inputs: {len(inputs1)} vs {len(inputs2)}"
            )

        result: List[TensorSpec] = []

        for t1, t2 in zip(inputs1, inputs2):
            if t1.dtype != t2.dtype:
                raise ValueError(f"Different types: {t1.dtype} vs {t2.dtype}")
            if len(t1.shape) != len(t2.shape):
                raise ValueError(
                    f"Different tensor sizes: {len(t1.shape)} vs {len(t2.shape)}"
                )
            shape: List[IntVar] = []
            for i, (d1, d2) in enumerate(zip(t1.shape, t2.shape)):
                if d1 == d2:
                    shape.append(IntImm(d1))
                else:
                    shape.append(IntVar([min(d1, d2), max(d1, d2)], f"dynamic_dim_{i}"))
            result.append(TensorSpec(shape, t1.dtype))

        return result

    @classmethod
    def gen_int_var_min_max(cls, vmin: int, vmax: int, name: str = None):  # noqa [B902]
        values = [vmin, vmax]
        if vmin == vmax:
            return IntImm(vmin, name=name)
        elif vmin < vmax:
            return IntVar([vmin, vmax], name=name)
        else:
            raise RuntimeError("Unsupported int var definition: {}".format(values))

    @classmethod
    def create_spec_from_int_vars(cls, int_vars: List[IntVar], dtype_list: torch.dtype):
        if len(int_vars) != len(dtype_list):
            raise ValueError(
                f"Different number of int_var and dtype_list: {len(int_vars)} vs {len(dtype_list)}"
            )
        res = []
        for int_var, dtype in zip(int_vars, dtype_list):
            res.append(TensorSpec(int_var, dtype))
        return res

    @classmethod
    def create_spec_from_shapes(
        cls, inputs_min: List[int], inputs_max: List[int], dtype_list: torch.dtype
    ) -> List["TensorSpec"]:
        if len(inputs_min) != len(inputs_max):
            raise ValueError(
                f"Different number of inputs: {len(inputs_min)} vs {len(inputs_max)}"
            )
        res = []
        for shape1, shape2, dtype in zip(inputs_min, inputs_max, dtype_list):
            if len(shape1) != len(shape2):
                raise ValueError(
                    f"Different number of input dims: {len(shape1)} vs {len(shape2)}"
                )

            shape: List[IntVar] = []
            for i, (d1, d2) in enumerate(zip(shape1, shape2)):
                if d1 == d2:
                    shape.append(IntImm(d1))
                else:
                    shape.append(IntVar([min(d1, d2), max(d1, d2)], f"dynamic_dim_{i}"))
            res.append(TensorSpec(shape, dtype))
        return res

    def to_random_tensor(self, use_lower_bound=True):
        shape = []
        for s in self.shape:
            if use_lower_bound:
                shape.append(s.lower_bound())
            else:
                shape.append(s.upper_bound())
        return torch.randn(shape).to(dtype=self.dtype)

    def to_specific_tensor(self, use_lower_bound, specify_num):
        shape = []
        for s in self.shape:
            if use_lower_bound:
                shape.append(s.lower_bound())
            else:
                shape.append(s.upper_bound())
        return torch.full(shape, specify_num).to(dtype=self.dtype)

    @classmethod
    def create_inputs_from_specs(
        cls, input_specs: List["TensorSpec"], use_lower_bound: bool, specify_num=None
    ) -> torch.Tensor:
        result = []
        for inp in input_specs:
            if specify_num is None:
                result.append(inp.to_random_tensor(use_lower_bound).cuda())
            else:
                result.append(
                    inp.to_specific_tensor(use_lower_bound, specify_num).cuda()
                )

        return result

    @classmethod
    def from_input_list_with_batch_size(
        cls, inputs: List[torch.Tensor], max_batch_size: int, batch_dim: int = 0
    ) -> List["TensorSpec"]:
        """
        Most of the recommendation models will work fine using this function.

        We make an assumption that inferred lowerable subgraph inputs will have
        a single batch dimension with the same max batch size.
        """
        result: List[TensorSpec] = []

        bs_dim = cls.find_batch_size_dim(inputs)
        for index, t in enumerate(inputs):
            shape: List[IntVar] = []
            for i, d in enumerate(t.shape):
                if i == bs_dim[index]:
                    shape.append(IntVar([1, max_batch_size], "batch_size"))
                else:
                    shape.append(IntImm(d))
            result.append(TensorSpec(shape, t.dtype))

        return result

    @classmethod
    def from_input_list_with_batch_size_jagged_tensor(
        cls,
        inputs: List[torch.Tensor],
        max_batch_size: int,
        max_batch_size_jagged_tensor: int,
        tag_val=None,
    ) -> List["TensorSpec"]:
        """
        Most of the recommendation models will work fine using this function.

        We make an assumption that inferred lowerable subgraph inputs will have
        a single batch dimension with the same max batch size.
        """
        result: List = []
        result_unsorted: List = []
        left_inputs: List = []
        left_inputs_ind: List = []
        for ind, t in enumerate(inputs):
            if t.shape[0] == tag_val:
                shape: List[IntVar] = []
                for i, d in enumerate(t.shape):
                    if i == 0:
                        shape.append(
                            IntVar(
                                [1, max_batch_size_jagged_tensor],
                                "batch_size_jagged_tensor",
                            )
                        )
                    else:
                        shape.append(IntImm(d))
                result_unsorted.append((ind, TensorSpec(shape, t.dtype)))
            else:
                left_inputs.append(t)
                left_inputs_ind.append(ind)

        bs_dim = cls.find_batch_size_dim(left_inputs)
        for index, t in enumerate(left_inputs):
            shape: List[IntVar] = []
            for i, d in enumerate(t.shape):
                if i == bs_dim[index]:
                    shape.append(IntVar([1, max_batch_size], "batch_size"))
                else:
                    shape.append(IntImm(d))
            result_unsorted.append((left_inputs_ind[index], TensorSpec(shape, t.dtype)))
        result = sorted(result_unsorted, key=lambda num: num[0])
        result = [r[1] for r in result]
        return result

    @classmethod
    # pyre-ignore [2]: Parameter `sample_input` must have a type other than `Any`
    def find_batch_size_dim(cls, inputs: Any) -> []:
        if isinstance(inputs, torch.Tensor) or len(inputs) <= 1:
            return [0]
        shapes = [i.shape for i in inputs]
        frequency_map = {}
        for shape in shapes:
            if len(shape) < 2:
                # By pass for rank-1 tensors. MRS model has rank-1 tensor carry no batch_size info
                continue
            # Dedup shape value for single tensor
            shape = set(shape)
            for i in shape:
                frequency_map[i] = frequency_map.get(i, 0) + 1
        sorted_frequency = sorted(frequency_map.items(), key=lambda x: -x[1])
        batch_size = sorted_frequency[0][0]

        bs_dim = []
        for i in inputs:
            # Default batch size dim = -1, indicate no batch_size
            dim = -1
            for index, val in enumerate(i.shape):
                if val == batch_size:
                    dim = index
                    break
            bs_dim.append(dim)

        return bs_dim

    @classmethod
    def from_input_list_with_batch_size_static_batch(
        cls, inputs: List[torch.Tensor], max_batch_size: int, batch_dim: int = 0
    ) -> List["TensorSpec"]:
        """
        Most of the recommendation models will work fine using this function.

        We make an assumption that inferred lowerable subgraph inputs will have
        a single batch dimension with the same max batch size.
        """
        result: List[TensorSpec] = []

        for t in inputs:
            shape: List[IntVar] = []
            for _, d in enumerate(t.shape):
                shape.append(IntImm(d))
            result.append(TensorSpec(shape, t.dtype))

        return result
