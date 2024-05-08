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
from typing import Any, Dict, List, Optional, Set, Union

import torch
from aitemplate.compiler.public import IntImm, IntVar

from .find_batch_size_dim import find_batch_size_dim as find_batch_size_dim_impl

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
        cls,
        inputs1: List[Union[torch.Tensor, List[torch.Tensor]]],
        inputs2: List[Union[torch.Tensor, List[torch.Tensor]]],
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
            if isinstance(t1, list):
                result.append(cls.from_two_input_lists(t1, t2))
                continue
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
    def from_two_input_lists_jagged_tensor(
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
        dynamic_dict = {}
        num_dynamic = 0
        for t1, t2 in zip(inputs1, inputs2):
            if t1.dtype != t2.dtype:
                raise ValueError(f"Different types: {t1.dtype} vs {t2.dtype}")
            if len(t1.shape) != len(t2.shape):
                raise ValueError(
                    f"Different tensor sizes: {len(t1.shape)} vs {len(t2.shape)}"
                )
            shape: List[IntVar] = []
            for _, (d1, d2) in enumerate(zip(t1.shape, t2.shape)):
                if d1 == d2:
                    shape.append(IntImm(d1))
                else:
                    dynamic_range = [min(d1, d2), max(d1, d2)]
                    tuple_range = tuple(dynamic_range)
                    dynamic_name = dynamic_dict.get(tuple_range, None)
                    # The rule here we record the dynamic range+dynamic name in a dict
                    # and extract it if it exists. If not, we will create the pair and append
                    # the pair to the dict
                    if dynamic_name:
                        shape.append(IntVar(dynamic_range, dynamic_name))
                    else:
                        if num_dynamic == 0:
                            shape.append(IntVar(dynamic_range, "batch_size"))
                            dynamic_dict[tuple_range] = "batch_size"
                        else:
                            shape.append(
                                IntVar(
                                    dynamic_range,
                                    f"batch_size_{num_dynamic}",
                                )
                            )
                            dynamic_dict[tuple_range] = f"batch_size_{num_dynamic}"
                        num_dynamic = num_dynamic + 1
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

    @staticmethod
    def _get_max_seq_lens_from_offsets(
        inputs: List[torch.Tensor],
        jagged_offsets_batch_dims: Set[int],
    ) -> Dict[int, int]:
        """
        Get the maximum sequence length encoded in each offsets tensor.

        Offsets tensors encode the length of each sequence in the corresponding
        jagged tensor. Here we extract the maximum sequence length in each offsets
        tensor in the inputs and associate it with the total_length (== offsets[-1])
        of the corresponding jagged tensor in the inputs.
        """
        offsets_inputs = [
            inp
            for inp in inputs
            # offsets tensors are rakn-1 and have a specific first dimension
            if len(inp.shape) == 1 and inp.shape[0] in jagged_offsets_batch_dims
        ]

        max_seq_lens = {}
        for offsets in offsets_inputs:
            offsets = offsets.cpu()
            # the last value in the offsets tensor is the total_length
            # dimension of the corresponding jagged tensor
            total_length = offsets[-1].item()
            # max. sequence length == max. consecutive offset difference
            max_seq_len = torch.max(offsets[1:] - offsets[:-1]).item()
            if total_length in max_seq_lens:
                # if multiple jagged tensors have the same total length,
                # set the max_seq_len to the maximum of all sequences
                # in all corresponding offsets tensors
                max_seq_lens[total_length] = max(
                    max_seq_lens[total_length], max_seq_len
                )
            else:
                max_seq_lens[total_length] = max_seq_len

            logger.info(
                f"Maximum sequence length {max_seq_lens[total_length]} "
                f"for the input jagged tensor with {total_length=} "
                "inferred from the offsets tensor."
            )

        return max_seq_lens

    @classmethod
    def try_getting_jagged_tensor_map(
        cls,
        inputs: List[torch.Tensor],
        jagged_tensor_batch_dims: Set[int],
        fx_inputs: Optional[List[torch.fx.Node]] = None,
    ) -> Optional[Dict[int, int]]:
        """
        Try getting a map associating each jagged tensor input with the offsets.

        In the jagged tensor operators, jagged tensors are normally passed as inputs
        together with the (one or more) offsets tensors. Here we try to infer the
        mapping between each jagged tensor input in the graph and the corresponding
        offsets tensor based on the pre-populated meta-parameters of the fx.Nodes
        corresponding to the jagged tensor inputs.
        """
        if fx_inputs is None or len(fx_inputs) != len(inputs):
            return None

        fx_input_shapes = []
        for fx_inp in fx_inputs:
            tensor_meta = fx_inp.meta.get("tensor_meta", None)
            if tensor_meta is None:
                return None
            shape = getattr(tensor_meta, "shape", None)
            if shape is None:
                return None
            fx_input_shapes.append(shape)

        for i, inp in enumerate(inputs):
            if inp.shape != fx_input_shapes[i]:
                # shape mismatch between the sample input
                # and the corresponding fx.Node in the graph
                return None

        jagged_tensor_map = {}
        seen_offsets_names = {}
        for i, inp in enumerate(inputs):
            if inp.shape[0] in jagged_tensor_batch_dims:
                offsets_name = fx_inputs[i].meta.get("offsets_name", None)
                if offsets_name is None:
                    # not a jagged tensor
                    continue
                if len(offsets_name) > 1:
                    # offsets name attached to the jagged tensor's
                    # fx.Node is either ambiguous: failing here
                    return None
                offsets_name = list(offsets_name)[0]
                if offsets_name not in seen_offsets_names:
                    seen_offsets_names[offsets_name] = len(seen_offsets_names)
                # associate the jagged tensor with the corresponding offsets
                jagged_tensor_map[i] = seen_offsets_names[offsets_name]

        return jagged_tensor_map

    @classmethod
    def from_input_list_with_batch_size_jagged_tensor(
        cls,
        inputs: List[torch.Tensor],
        max_batch_size: int,
        max_sequence_length: int,
        jagged_tensor_batch_dims: Set[int],
        jagged_offsets_batch_dims: Set[int],
        additional_inputs: List[torch.Tensor] = None,
        infer_max_seq_lens_from_offsets: bool = False,
        fx_inputs: List[torch.fx.Node] = None,
        jagged_tensor_map: Optional[Dict[int, int]] = None,
    ) -> List["TensorSpec"]:
        """
        Most of the recommendation models will work fine using this function.

        We make an assumption that inferred lowerable subgraph inputs will have
        a single batch dimension with the same max batch size.
        """
        max_seq_lens_from_offsets = {}
        if infer_max_seq_lens_from_offsets:
            max_seq_lens_from_offsets = cls._get_max_seq_lens_from_offsets(
                inputs=inputs,
                jagged_offsets_batch_dims=jagged_offsets_batch_dims,
            )

        result: List = []
        result_unsorted: List = []
        left_inputs: List = []
        left_inputs_ind: List = []
        left_additional_inputs: List = []
        for ind, t in enumerate(inputs):
            batch_dim: int = t.shape[0]
            batch_dim_lower_bound: int = 0
            batch_dim_upper_bound: int = 0
            batch_dim_name: str = ""
            if jagged_tensor_map and ind in jagged_tensor_map:
                batch_dim_lower_bound = 0  # when all sequences are empty
                # if the maximum sequence length for this jagged tensor was not
                # inferred from the offsets, we use the globally configured
                # max_sequence_length (passed as argument to this function)
                max_seq_len = max_seq_lens_from_offsets.get(
                    batch_dim, max_sequence_length
                )
                batch_dim_upper_bound = max_batch_size * max_seq_len
                batch_dim_name = f"batch_size_jagged_tensor_id_{jagged_tensor_map[ind]}"
            elif not jagged_tensor_map and batch_dim in jagged_tensor_batch_dims:
                batch_dim_lower_bound = 0
                max_seq_len = max_seq_lens_from_offsets.get(
                    batch_dim, max_sequence_length
                )
                batch_dim_upper_bound = max_batch_size * max_seq_len
                batch_dim_name = f"batch_size_jagged_tensor_{batch_dim}"
            elif batch_dim in jagged_offsets_batch_dims:
                batch_dim_lower_bound = 2  # prefix 0 + at least one offset
                batch_dim_upper_bound = max_batch_size + 1
                batch_dim_name = f"batch_size_jagged_offsets_{batch_dim}"

            if batch_dim_upper_bound > 0:
                shape: List[IntVar] = []
                for i, d in enumerate(t.shape):
                    if i == 0:
                        shape.append(
                            IntVar(
                                values=[
                                    batch_dim_lower_bound,
                                    batch_dim_upper_bound,
                                ],
                                name=batch_dim_name,
                            )
                        )
                    else:
                        shape.append(IntImm(d))
                result_unsorted.append((ind, TensorSpec(shape, t.dtype)))
            else:
                left_inputs.append(t)
                left_inputs_ind.append(ind)

        if additional_inputs:
            for ind in left_inputs_ind:
                left_additional_inputs.append(additional_inputs[ind])
            input_specs_left = TensorSpec.from_two_input_lists_jagged_tensor(
                left_inputs, left_additional_inputs
            )
            assert len(input_specs_left) == len(
                left_inputs_ind
            ), "Unexpected length for left inputs"

            for index, ind_value in enumerate(left_inputs_ind):
                result_unsorted.append((ind_value, input_specs_left[index]))

        else:
            bs_dim = cls.find_batch_size_dim(left_inputs)
            for index, t in enumerate(left_inputs):
                shape: List[IntVar] = []
                for i, d in enumerate(t.shape):
                    if i == bs_dim[index]:
                        shape.append(IntVar([1, max_batch_size], "batch_size"))
                    else:
                        shape.append(IntImm(d))
                result_unsorted.append(
                    (left_inputs_ind[index], TensorSpec(shape, t.dtype))
                )
        result = sorted(result_unsorted, key=lambda num: num[0])
        result = [r[1] for r in result]
        return result

    @classmethod
    # pyre-ignore [2]: Parameter `sample_input` must have a type other than `Any`
    def find_batch_size_dim(
        cls,
        inputs: Any,
        can_non_first_dim_be_dynamic: bool = True,
        can_dim_value_one_be_dynamic: bool = True,
        # pyre-fixme Invalid type [31]
    ) -> []:
        return find_batch_size_dim_impl(
            inputs, can_non_first_dim_be_dynamic, can_dim_value_one_be_dynamic
        )

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
