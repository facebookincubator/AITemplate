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
from typing import Any

import torch


def find_batch_size_dim(
    inputs: Any,
    can_non_first_dim_be_dynamic: bool = True,
    can_dim_value_one_be_dynamic: bool = True,
    # pyre-fixme Invalid type [31]
) -> []:
    if isinstance(inputs, torch.Tensor) or len(inputs) <= 1:
        return [0]
    shapes = [i.shape for i in inputs]
    frequency_map = {}
    position_scores = {}
    first_dims = set()
    for shape in shapes:
        if len(shape) < 2:
            # By pass for rank-1 tensors. MRS model has rank-1 tensor carry no batch_size info
            continue
        # Dedup shape value for single tensor
        first_dims.add(shape[0])
        seen_dims = set()
        valid_len = len(shape) if can_non_first_dim_be_dynamic else 1
        for i in range(valid_len):
            dim = shape[i]
            if dim not in seen_dims:
                frequency_map[dim] = frequency_map.get(dim, 0) + 1
                position_scores[dim] = position_scores.get(dim, 0) + i
                seen_dims.add(dim)

    if len(first_dims) == 1:
        # first dim is the same in every input: we use it as batch_size
        batch_size = first_dims.pop()
    elif frequency_map:
        # first dims are different: we use the most frequent dim as batch_size
        # if there is more than 1 most frequent dim, we choose the one with the
        # lowest position score (i.e., the leftmost of the most frequent ones)
        sorted_frequency = sorted(
            frequency_map.items(),
            key=lambda x: (-x[1], position_scores[x[0]]),
        )
        if len(sorted_frequency) > 1:
            if not can_dim_value_one_be_dynamic and sorted_frequency[0][0] == 1:
                # It's often that dim value one indicates a non-dynamic dimension.
                # If the user says so, we pick the second most frequent value.
                batch_size = sorted_frequency[1][0]
            else:
                batch_size = sorted_frequency[0][0]
        else:
            if not can_dim_value_one_be_dynamic and sorted_frequency[0][0] == 1:
                batch_size = -1
            else:
                batch_size = sorted_frequency[0][0]
    else:
        # no dims to sort: no batch_size
        batch_size = -1

    bs_dim = []
    for i in inputs:
        # Default batch size dim = -1, indicate no batch_size
        dim = -1
        for index, val in enumerate(i.shape):
            if not can_non_first_dim_be_dynamic and index > 0:
                break
            if val == batch_size:
                dim = index
                break
        bs_dim.append(dim)

    return bs_dim
