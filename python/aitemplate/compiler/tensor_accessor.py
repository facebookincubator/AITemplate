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
TensorAccessor definition.
"""

import copy
import logging

# pylint: disable=C0103,C0301,W0612

from pprint import pformat
from typing import Any, Iterable, List, Optional

from aitemplate.compiler.base import IntImm, IntVar, Tensor


_LOGGER = logging.getLogger(__name__)


class TensorAccessor:
    """
    A tensor accessor which manages how to access a Tensor.
    Must always be used together with a Tensor.
    """

    def __init__(self, original_tensor: Tensor) -> None:
        """
        Initializes the TensorAccessor with an "original" tensor.
        """
        super().__init__()
        # Tensor offset in terms of number of elements compared to the base tensor.
        self.offset = 0
        self.original_shapes = original_tensor._attrs["shape"]
        # We need dtype for computing alignment requirement
        self.tensor_dtype = original_tensor.dtype()
        # This strictly means that the tensor's memory itself is contiguous
        self.is_contiguous = True

        # These variables are only set when self.stride_dim != None.
        # A tensor can be contiguous and still come from a strided tensor,
        # e.g., when stride_dim == 0
        self.is_from_strided_tensor = False
        self.stride_dim = None
        self.actual_shapes = None

        # Total number of elements starting from the stride dim, before & after.
        self.original_total_elements_from_stride_dim = None
        self.actual_total_elements_from_stride_dim = None

        # List[(original_dim_group, actual_dim_group)], where
        # original_dim_group is a list of indices for self.original_shapes, and
        # actual_dim_group is a list of indices for self.actual_shapes.
        #
        # This field is used to record dim mapping between self.original_shapes
        # and self.actual_shapes when update_base_tensor_shape() is called,
        # which means this TensorAccessor is used to handle a view op (e.g. reshape).
        # This will be used together with strided_dim to calculate
        # strides based on original tensor shape and dim.
        #
        # By default, self._dim_mapping = [([0], [0]), ([1], [1]), ...., ([n], [n])],
        # where n = len(self.original_shapes) - 1.
        # It represents that there is 1:1 dim mapping relationship between
        # self.original_shapes and self.actual_shapes.
        #
        # When self.update_base_tensor_shape() is called, # dimension may be different
        # between self.original_shapes and self.actual_shapes.
        # e.g. The original tensor is in shape [2, 3, 2], and it's reshaped to [2, 6].
        # In this case, self._dim_mapping = [([0], [0]), ([1, 2], [1])], which represents
        # that self.original_shapes[0] and self.actual_shapes[0] are in the same group,
        # and self.original_shapes[1:2] and self.actual_shapes[1] are in the same group.
        #
        # It's possible that such a mapping cannot be calculated (e.g. because of
        # dynamic shapes), and in such a case, self._dim_mapping is updated to None
        # to represent that this tensor_accessor cannot be further updated with
        # another stride dims.
        self._dim_mapping = [([i], [i]) for i in range(len(self.original_shapes))]

    def __deepcopy__(self, memo):
        res = copy.copy(self)
        res.original_shapes = copy.deepcopy(self.original_shapes, memo)
        res.actual_shapes = copy.deepcopy(self.actual_shapes, memo)
        res._dim_mapping = copy.deepcopy(self._dim_mapping, memo)
        return res

    def __str__(self) -> str:
        return pformat(vars(self), indent=2)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorAccessor):
            return False
        attrs = vars(self)
        for attr in attrs:
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr)
            if self_attr != other_attr:
                return False
        return True

    def _try_gen_dim_mapping(self):
        """
        Updates self._dim_mapping based on self.original_shapes and self.actual_shapes.
        Check comments for self._dim_mapping for more info.
        """

        # Set self._dim_mapping = None.
        self._dim_mapping = None
        dim_mapping = []

        original_value = None
        actual_value = None
        original_idx = 0
        actual_idx = 0
        prev_original_idx = 0
        prev_actual_idx = 0

        # Add a dummy entry at the end of original_shapes / actual_shapes to
        # make boundary checks easier.
        INT_MAX = int("0x" + "F" * 16, 16)
        original_shapes = list(self.original_shapes) + [IntImm(INT_MAX)]
        actual_shapes = list(self.actual_shapes) + [IntImm(INT_MAX)]

        while original_idx < len(original_shapes) and actual_idx < len(actual_shapes):
            original_d = original_shapes[original_idx]
            actual_d = actual_shapes[actual_idx]

            if original_d == IntImm(1) and original_value is None:
                # Create separate groups for "1"s.
                original_idx += 1
                dim_mapping.append((list(range(prev_original_idx, original_idx)), []))
                prev_original_idx = original_idx
                continue

            if actual_d == IntImm(1) and actual_value is None:
                # Create separate groups for "1"s.
                actual_idx += 1
                dim_mapping.append(([], list(range(prev_actual_idx, actual_idx))))
                prev_actual_idx = actual_idx
                continue

            if not isinstance(original_d, IntImm) or not isinstance(actual_d, IntImm):
                # If there are dynamic dims, check whether these two dynamic dims are the same.
                # Only allow one dynamic dim per group.
                if (
                    original_d != actual_d
                    or original_value is not None
                    or actual_value is not None
                ):
                    return
                else:
                    original_value = original_d
                    actual_value = actual_d
                    original_idx += 1
                    actual_idx += 1
            elif original_value is None or actual_value is None:
                # Assign values to original_value / actual_value first.
                if original_value is None:
                    original_value = original_d.value()
                    original_idx += 1
                if actual_value is None:
                    actual_value = actual_d.value()
                    actual_idx += 1
            else:
                # Choose to advance original_idx or actual_idx based on
                # original_value / actual_value comparisons.
                if original_value < actual_value:
                    original_value *= original_d.value()
                    original_idx += 1
                elif original_value > actual_value:
                    actual_value *= actual_d.value()
                    actual_idx += 1
                else:
                    raise AssertionError("This branch should never be reached.")

            if original_value == actual_value:
                dim_mapping.append(
                    (
                        list(range(prev_original_idx, original_idx)),
                        list(range(prev_actual_idx, actual_idx)),
                    )
                )
                prev_original_idx = original_idx
                prev_actual_idx = actual_idx
                original_value = None
                actual_value = None

        if (
            original_value is not None
            or actual_value is not None
            or original_idx != len(original_shapes)
            or actual_idx != len(actual_shapes)
        ):
            _LOGGER.debug(f"tail processing failed, dim_mapping: {dim_mapping}")
            return

        # Remove the last dummy group.
        dim_mapping = dim_mapping[:-1]

        # Assign new dim_mapping to self._dim_mapping.
        self._dim_mapping = dim_mapping
        _LOGGER.debug(f"generate dim_mapping: {dim_mapping}")

    def try_get_stride_strs(
        self, dim: int, dim_names: List[str] = None
    ) -> Optional[List[str]]:
        """
        Tries to return a list of stride strs for the given dim.
        Note that both dim and dim_names are based on self.original_shapes.

        Returns None if this function fails to calculate stride strs.
        """

        assert dim < len(self.original_shapes), (
            f"dim {dim} must be smaller than original_shapes' rank, "
            f"{len(self.original_shapes)}"
        )
        if dim_names is not None:
            assert len(dim_names) == len(self.original_shapes), (
                "dim_names must have the same length as shapes, "
                f"dim_names: {dim_names}, shapes: {self.original_shapes}"
            )

        def _get_value_or_names(
            shape: List[IntVar], indices: Iterable[int]
        ) -> List[str]:
            res = []
            for index in indices:
                d = shape[index]
                if isinstance(d, IntImm):
                    res.append(str(d.value()))
                else:
                    res.append(
                        dim_names[index] if dim_names is not None else d._attrs["name"]
                    )
            return res

        if self.stride_dim is None:
            # There are no stride dims. Use original shapes to calculate directly.
            return _get_value_or_names(
                self.original_shapes, range(dim + 1, len(self.original_shapes))
            )

        if self._dim_mapping is None:
            # self._dim_mapping cannot be generated successfully.
            # Return None to represent an error.
            _LOGGER.debug("Failed to get dim mapping.")
            return None

        # Loop through self._dim_mapping to generate stride_strs.
        found_original_dim_group = False
        res = []
        for original_group, actual_group in self._dim_mapping:
            if not found_original_dim_group:
                if dim in original_group:
                    found_original_dim_group = True
                    idx = original_group.index(dim)
                    if (self.stride_dim in actual_group) and (
                        idx != len(original_group) - 1
                    ):
                        # If self.stride_dim is in the corresponding group,
                        # need to make sure that dim is the last dim
                        # inside the original group.
                        # Otherwise, we cannot compute strides.
                        _LOGGER.debug(
                            "Multiple dims in stride_dim group. "
                            f"dim_mapping: {self._dim_mapping}, "
                            f"dim: {dim}, stride_dim: {self.stride_dim}, self: {self}"
                        )
                        return None
                    res.extend(
                        _get_value_or_names(
                            self.original_shapes, original_group[idx + 1 :]
                        )
                    )
            else:
                if self.stride_dim in actual_group:
                    if actual_group.index(self.stride_dim) != 0:
                        _LOGGER.debug(
                            f"Stride dim {self.stride_dim} is not the first dim "
                            f"of the underlying group {actual_group}."
                        )
                        return None
                    res.extend(_get_value_or_names(self.actual_shapes, actual_group))
                else:
                    res.extend(
                        _get_value_or_names(self.original_shapes, original_group)
                    )

        _LOGGER.debug(
            f"dim: {dim}, stride_dim: {self.stride_dim}, "
            f"mapping: {self._dim_mapping}, stride_strs: {res}, "
            f"original: {self.original_shapes}, actual: {self.actual_shapes}"
        )
        return res

    def stride(self, dim: int) -> int:
        """
        Returns stride (a number) for the given dim.
        Note that dim is based on self.original_shapes.
        This API assumes that all dims after dim are static (IntImm).

        Throws RuntimeError if such a stride number cannot be computed.
        """

        for i, d in enumerate(self.original_shapes[dim + 1 :], dim + 1):
            if not isinstance(d, IntImm):
                raise RuntimeError(
                    f"Can only calculate static stride from static dim: {i}, "
                    f"original shapes: {self.original_shapes}."
                )
        strides = self.try_get_stride_strs(dim)
        if strides is None:
            raise RuntimeError("Failed to get stride strs!")
        stride = 1
        for s in strides:
            stride *= int(s)
        return stride

    def gen_stride_str(self, dim: int, dim_names: List[str]) -> str:
        """
        Returns the str to calculate the stride of a certain dim. This is
        a temporary solution to get around dynamic shapes problems with
        tensor_accessor. dim_names is a list of str, such as ["B", "M", "K"]
        for the first input to bmm_rcr.

        Note that both dim and dim_names are based on self.original_shapes.

        Throws RuntimeError if such a stride number cannot be computed.
        """
        strides = self.try_get_stride_strs(dim, dim_names)
        if strides is None:
            raise RuntimeError("Failed to get stride strs!")
        return " * ".join(strides)

    def update_base_tensor(
        self, new_tensor: Tensor, stride_dim: int, stride_dim_offset: int
    ) -> None:
        """
        Updates the TensorAccessor with a new base tensor.
        This API is useful to handle ops with a stride dim, e.g. split, cat.
        It can also be used by slice if slice is only operated on one dim.
        """

        assert (
            self.stride_dim is None
        ), "Tensor accessor cannot be updated once stride_dim is set!"

        original_shapes = (
            self.actual_shapes
            if self.actual_shapes is not None
            else self.original_shapes
        )
        self.actual_shapes = new_tensor._attrs["shape"]
        self.stride_dim = stride_dim
        assert len(self.actual_shapes) == len(original_shapes), (
            f"Original tensor and new tensor must have the same number of dims! "
            f"Original tensor shape: {original_shapes}, new tensor shape: {self.actual_shapes}"
        )
        assert (
            len(self.actual_shapes) > stride_dim
        ), f"stride_dim {stride_dim} must be less than #dims {len(self.actual_shapes)}!"
        assert isinstance(
            original_shapes[stride_dim], IntImm
        ), "Stride dim can't be dynamic!"
        assert isinstance(
            self.actual_shapes[stride_dim], IntImm
        ), "Stride dim can't be dynamic!"

        self.original_total_elements_from_stride_dim = original_shapes[
            stride_dim
        ].value()
        self.actual_total_elements_from_stride_dim = self.actual_shapes[
            stride_dim
        ].value()
        self.offset = stride_dim_offset

        for original_shape, actual_shape in zip(
            original_shapes[stride_dim + 1 :], self.actual_shapes[stride_dim + 1 :]
        ):
            assert isinstance(
                original_shape, IntImm
            ), "Dims after the stride dim must have static shapes! Shapes: {}".format(
                original_shapes
            )
            assert isinstance(
                actual_shape, IntImm
            ), "Dims after the stride dim must have static shapes! Shapes: {}".format(
                self.actual_shapes
            )
            assert (
                original_shape._attrs["values"] == actual_shape._attrs["values"]
            ), "original shapes {} and actual shapes {} after the stride dim must be equal! ".format(
                original_shapes, self.actual_shapes
            )
            value = actual_shape._attrs["values"][0]
            self.original_total_elements_from_stride_dim *= value
            self.actual_total_elements_from_stride_dim *= value
            self.offset *= value

        if (
            stride_dim > 0
            and self.actual_total_elements_from_stride_dim
            > self.original_total_elements_from_stride_dim
        ):
            self.is_contiguous = False

        if (
            self.actual_total_elements_from_stride_dim
            > self.original_total_elements_from_stride_dim
        ):
            self.is_from_strided_tensor = True

    def update_base_tensor_shape(self, new_tensor: Tensor) -> None:
        """
        Updates the TensorAccessor's actual shape.
        This API is useful to handle view ops, e.g. reshape, flatten, etc.
        """

        assert (
            self.stride_dim is None
        ), "Tensor accessor cannot be updated once stride_dim is set!"

        self.actual_shapes = new_tensor._attrs["shape"]
        original_dynamic_dims = {
            dim for dim in self.original_shapes if not isinstance(dim, IntImm)
        }
        new_dynamic_dims = {
            dim for dim in self.actual_shapes if not isinstance(dim, IntImm)
        }
        assert original_dynamic_dims == new_dynamic_dims, (
            "Original tensor and actual tensor have different dynamic dimensions! "
            f"Original tensor: {self.original_shapes}, "
            f"actual tensor: {self.actual_shapes}!"
        )
        self._try_gen_dim_mapping()

    def is_rightmost_dim_contiguous(self, cat_dim: int) -> bool:
        """Check if the rightmost diminsion would be contiguous after
        concatenation along a given cat_dim. This is a necessary condition for
        GEMM+concat fusion, since GEMM doesn't support discontinuous rightmost
        dimension for row-major outout. Rightmost diminsion is contiguous iff
        the concat dimension corresponds to one of the dimensions in the
        original shape and it's the first dimension in its group of actual
        dimensions.
        """
        num_groups = len(self._dim_mapping)
        for group_idx in range(num_groups):
            original_group, actual_group = self._dim_mapping[group_idx]
            if cat_dim in actual_group:
                if actual_group.index(cat_dim):
                    # Concat dimension isn't the first in its group
                    return False
                # Check that there is at least one non-empty original group to the
                # right of the group where cat_dim found (inclusive)
                while (group_idx < num_groups) and not len(
                    self._dim_mapping[group_idx][0]
                ):
                    group_idx += 1
                if group_idx >= num_groups:
                    # There are no original dimensions to the right (inclusive) of concat
                    # dimension. Concat dimension is an unsqueezed dimension at the end
                    # of the shape, fusion is impossible.
                    return False
                return True
        return False
