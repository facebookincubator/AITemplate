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
import random
from itertools import product
from typing import List, Tuple

import torch

from aitemplate.testing.test_utils import get_torch_full_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype, torch_dtype_to_string


def _check_offsets(
    offsets_list: List[List[int]],
) -> None:
    offsets_len = len(offsets_list[0])
    for offsets in offsets_list:
        assert offsets[0] == 0
        assert len(offsets) == offsets_len
        for j in range(1, len(offsets)):
            assert offsets[j] >= offsets[j - 1]
        offsets_len = offsets[-1] + 1


def _get_preceding_offset_idx(
    idx: int,
    offsets: List[int],
) -> Tuple[int, int]:
    result = None
    left, right = 0, len(offsets) - 1
    while left <= right:
        mid = (left + right) // 2
        offset = offsets[mid]
        if offset <= idx:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result, offsets[result]


def _jagged_idx_to_dense_idx(
    jagged_idx: int,
    offsets_list: List[List[int]],
) -> List[int]:
    assert jagged_idx < offsets_list[-1][-1]

    result = []
    for offsets in reversed(offsets_list):
        offset_idx, offset = _get_preceding_offset_idx(
            idx=jagged_idx,
            offsets=offsets,
        )
        result.append(jagged_idx - offset)
        jagged_idx = offset_idx
    result.append(jagged_idx)

    return list(reversed(result))


def jagged_to_dense(
    jagged: torch.Tensor,
    offsets_list: List[torch.Tensor],
    dense_shape: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Convert a jagged Tensor (with the offsets) into a dense Tensor.

    The function converts a jagged Tensor (and the offsets) to
    a rectangular dense Tensor, using the padding_value at the
    positions of the resulting dense Tensor where the input
    jagged Tensor doesn't have elements.

    Parameters
    ----------
    jagged : torch.Tensor
        The jagged Tensor with the shape `[total_length, D1, ..., Dm]`,
        The first dimension, `total_length`, encodes the `batch_dim` and
        the jagged dims of the jagged Tensor. The following m dimensions,
        `D1, ..., Dm`, are regular dense dimensions following the jagged
        dimensions of the jagged Tensor.

    offsets_list : List[torch.Tensor]
        A list of rank-1 Tensors, each representing the offsets along one
        of the jagged dimensions of the jagged Tensor. The number of offsets
        Tensors in the list must correspond to the number of jagged dimensions
        encoded in the first `total_length` dimension of `jagged`. The offsets
        Tensors must be consistent with the offset specification:

            - batch_dim == len(offsets[0]) - 1
            - offsets[i][-1] == len(offsets[i+1])) - 1
            - offsets[-1][-1] == total_length

    dense_shape : List[int]
        The shape of the resulting dense Tensor. The last m dimensions in
        the `dense_shape` must be equal to `[D1, ..., Dm]` in the jagged
        Tensor shape. The first dimension must be the `batch_dim`. The
        following n dimensions must correspond to the n jagged dimensions
        of the jagged Tensor, with the values equal to the maximum possible
        values of the jagged dimensions.

    padding_value : float
        The value to fill the dense Tensor with at the positions where
        there are no elements in the jagged Tensor. Default: 0.0.

    Returns
    -------
    torch.Tensor
        The dense tensor with the `dense_shape` converted from the
        `jagged` Tensor, with the `padding_value` at other positions.
    """
    assert all(t.dim() == 1 for t in offsets_list)
    offsets_list = [list(t.cpu().numpy()) for t in offsets_list]

    _check_offsets(offsets_list)
    assert len(dense_shape) - len(jagged.shape) == len(offsets_list)
    assert jagged.shape[1:] == tuple(dense_shape[1 + len(offsets_list) :])
    for i, offsets in enumerate(offsets_list):
        dense_dim = dense_shape[i + 1]
        for j in range(1, len(offsets)):
            assert offsets[j] - offsets[j - 1] <= dense_dim

    dtype = torch_dtype_to_string(jagged.dtype)
    result = get_torch_full_tensor(
        shape=dense_shape,
        fill_value=padding_value,
        dtype=dtype,
    )

    total_length = jagged.shape[0]
    for jagged_idx in range(total_length):
        dense_idx = _jagged_idx_to_dense_idx(
            jagged_idx=jagged_idx,
            offsets_list=offsets_list,
        )
        result[tuple(dense_idx)] = jagged[jagged_idx]

    return result


def _dense_idx_to_jagged_idx(
    dense_idx: List[int],
    offsets_list: List[List[int]],
) -> int:
    assert len(dense_idx) == 1 + len(offsets_list)

    offset = 0
    for i, (d, offsets) in enumerate(zip(dense_idx, offsets_list)):
        prev_offset, next_offset = offsets[offset + d : offset + d + 2]
        group_size = next_offset - prev_offset
        if dense_idx[i + 1] >= group_size:
            return -1
        offset = prev_offset
    offset += dense_idx[-1]

    return offset


def dense_to_jagged(
    dense: torch.Tensor,
    offsets_list: List[torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Convert a dense Tensor into a jagged Tensor (using the offsets).

    The function converts a rectangular dense Tensor to a compactly
    represented subset of its values: a jagged Tensor, using the offsets.
    The padding_value is used at the positions of the resulting jagged
    Tensor where the input dense Tensor doesn't have elements.

    Parameters
    ----------
    dense : torch.Tensor
        A Tensor with the shape `[batch_dim, N1, ..., Nn, D1, ..., Dm]`.
        The first n+1 dimensions of the dense Tensor are encoded into
        the first `total_length` dimension of the resulting jagged
        Tensor, using the specified offsets. Importantly, the values in
        the dense Tensor outside of what the offsets specify are omitted
        in the resulting jagged Tensor.

    offsets_list : List[torch.Tensor]
        A list of rank-1 Tensors, each representing the offsets along one
        of the jagged dimensions of the jagged Tensor. The number of offsets
        Tensors in the list must correspond to the number of jagged dimensions
        encoded in the first `total_length` dimension of the resulting jagged
        Tensor. The offsets Tensors must be consistent with the offset
        specification:

            - batch_dim == len(offsets[0]) - 1
            - offsets[i][-1] == len(offsets[i+1])) - 1
            - offsets[-1][-1] == total_length

    padding_value : float
        The value to fill the jagged Tensor with at the positions where
        there are no elements in the dense Tensor (e.g., the consecutive
        offset difference is longer than the corresponding N dimension
        in the dense Tensor input). Default: 0.0.

    Returns
    -------
    torch.Tensor
        The jagged tensor converted from the `dense` Tensor using
        the offsets, with the `padding_value` at the positions
        not available in the `dense` Tensor.
    """
    assert all(t.dim() == 1 for t in offsets_list)
    offsets_list = [list(t.cpu().numpy()) for t in offsets_list]

    _check_offsets(offsets_list)
    assert len(offsets_list) < len(dense.shape)

    total_length = offsets_list[-1][-1]
    inner_shape = dense.shape[1 + len(offsets_list) :]
    jagged_shape = [total_length, *inner_shape]

    dtype = torch_dtype_to_string(dense.dtype)
    result = get_torch_full_tensor(
        shape=jagged_shape,
        fill_value=padding_value,
        dtype=dtype,
    )

    for dense_idx in product(*[range(d) for d in dense.shape[: 1 + len(offsets_list)]]):
        jagged_idx = _dense_idx_to_jagged_idx(
            dense_idx=dense_idx,
            offsets_list=offsets_list,
        )
        if jagged_idx != -1:
            result[jagged_idx] = dense[tuple(dense_idx)]

    return result


def generate_offsets(
    batch_size: int,
    max_seq_len: int,
    load_factor: float,
    offsets_dtype: str,
    spread_radius: float = 0.1,
) -> torch.Tensor:
    """
    Generate a rank-1 Tensor of offsets for the given load factor.

    This function generates a single linear offset Tensor for a
    single jagged dimension in a jagged Tensor with the batch_dim
    equal to `batch_size` and maximum value along the jagged
    dimension equal to `max_seq_len`. The `load_factor` in [0, 1]
    specifies how "full" should the jagged Tensor described by
    the resulting offsets should be, compared to the corresponding
    dense Tensor with a rectangular shape [batch_size, max_seq_len,
    D1, ..., Dm]. The offset differences (== the lengths along the
    jagged dimensions) are sampled randomly, to arrive close (but not
    necessarily equal) to the specified `load_factor` in total.

    When sampled out of the [0, N] interval, the offset differences
    are clamed to stay within the [0, N] interval.

    Parameters
    ----------
    batch_size : int
        The batch_dim of the jagged Tensor specified by the offsets.
    max_seq_len : int
        The maximum length along the jagged dimension specified by
        the offsets.
    load_factor : float
        The fraction of the [batch_size, max_seq_len, D1, ..., Dm]-
        shaped dense Tensor that the total (compactly represented)
        jagged Tensor data should correspond to.
    offsets_dtype : str
        The type of the resulting offsets Tensor.
    spread_radius : float
        The radius of the spread around int(max_seq_len * load_factor)
        that the offset differences should be randomly sampled from.
        Default: 0.1.

    Returns
    -------
    torch.Tensor
        The resulting rank-1 Tensor of offsets.
    """
    assert 0 <= load_factor <= 1
    assert 0 <= spread_radius <= 1

    if load_factor < 1:
        spread = int(max_seq_len * spread_radius)
        mean = int(max_seq_len * load_factor)
        lengths = [
            mean + random.randint(-spread, spread + 1) for _ in range(batch_size)
        ]
        lengths = [max(min(L, max_seq_len), 0) for L in lengths]
    else:
        lengths = [max_seq_len] * batch_size

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
    return torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()


def batched_dense_vec_jagged_2d_mul_ref(
    vectors: torch.Tensor,  # [B, H, N]
    matrices: torch.Tensor,  # [sum_B(N_B), H, D]
    offsets: torch.Tensor,  # [B + 1]
):
    """
    Reference function for fbgemm batched_dense_vec_jagged_2d_mul.
    https://pytorch.org/FBGEMM/python-api/jagged_tensor_ops.html#torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul

    Parameters
    ----------
    vecrors: torch.Tensor
        Batch of vectors of the shape [B, H, N]. N is the maximum
        sequence length in the jagged Tensor `matrices`. Each vector
        in the batch is N-sized. The effective batch size is B * H.
    matrices: torch.Tensor
        Batch of jagged matrices (in a jagged Tensor) of the shape
        [sum_B(N_B), H, D]. The first dimension encodes the batch
        B of sequneces of variable length: from 0 to N. The matrices
        have variable number of rows (determined by the variable
        sequence lengths) and fixed number of columns: D. H is a
        factor of the effective batch size, just pulled to the
        right of the sum_B(N_B) dimension.
    offsets: torch.Tensor
        Rank-1 offsets Tensor describing the single jagged dimension
        (from 0 to N) in the jagged `matrices`.

    Returns
    -------
    torch.Tensor
        Batch of vectors resulting from the batched vector x jagged
        matrix multiplication. Shape: [B, H, D] (as N in the `vectors`
        is contracted with the variable sequence length encoded in the
        sum_B(N_B) dimension of the `matrices`).
    """
    assert vectors.dim() == 3
    B, H, N = vectors.size()

    assert matrices.dim() == 3
    assert matrices.size(1) == H
    D = matrices.size(2)

    assert offsets.dim() == 1
    assert offsets.size(0) == B + 1

    # pad the jagged matrices with zeros
    padded_matrices = jagged_to_dense(
        jagged=matrices,
        offsets_list=[offsets],
        dense_shape=[B, N, H, D],
        padding_value=0.0,
    )  # [B, N, H, D]

    return torch.matmul(
        vectors.unsqueeze(dim=2),  # [B, H, 1, N]
        padded_matrices.permute([0, 2, 1, 3]),  # [B, H, N, D]
    ).squeeze(
        dim=2
    )  # [B, H, D]


def add_jagged_dense_ref(
    jagged: torch.Tensor,
    offsets_list: List[torch.Tensor],
    dense: torch.Tensor,
    jagged_max_shape: List[int] = None,
) -> torch.Tensor:
    """The reference function for jagged / dense elementwise add."""
    if jagged_max_shape is None:
        jagged_max_shape = dense.shape

    assert len(jagged.shape) + len(offsets_list) >= len(dense.shape)
    assert len(jagged_max_shape) == len(jagged.shape) + len(offsets_list)

    return dense_to_jagged(
        dense=(
            dense
            + jagged_to_dense(
                jagged=jagged,
                offsets_list=offsets_list,
                dense_shape=jagged_max_shape,
                padding_value=0.0,
            )
        ),
        offsets_list=offsets_list,
        padding_value=-1.0,
    )
