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
Base operator definition for reduce-family ops.
"""
import itertools
import logging

from typing import List

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.dtype import get_dtype_size
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.utils import shape_utils
from aitemplate.utils.tensor_utils import wrap_dim

# pylint: disable=C0103,W0221


_LOGGER = logging.getLogger(__name__)


class reduce_base(Operator):
    """The base class for reduce ops."""

    def __init__(self, dim, keepdim=False, dtype=None) -> None:
        """
        Parameters
        ----------
        dim : int or tuple of python:ints
            the dimension or dimensions to reduce
        keepdim : bool
            keep the reduced dimensions if True, default is False
        dtype : optional str
            the type of the return tensor. If it is not None,
            the input tensor is casted to dtype before reduction.
        Raises
        ------
        RuntimeError : duplicate values in the dim list
        """
        super().__init__()
        if isinstance(dim, int):
            dim = [dim]
        elif isinstance(dim, (list, tuple)):
            if not all(isinstance(x, int) for x in dim):
                raise RuntimeError("dim must be either int or a list/tuple of ints.")
            dim = list(dim)
        else:
            raise RuntimeError("dim must be either int or a list/tuple of ints.")
        dup_dims = {d for d in dim if dim.count(dim) > 1}
        if len(dup_dims) > 1:
            raise RuntimeError(
                "dim {d} appears multiple times in the list of dims".format(
                    d=dup_dims[0]
                )
            )
        self._attrs["op"] = "reduce"
        self._attrs["reduction_axes"] = dim
        self._attrs["keepdim"] = keepdim
        self._attrs["output_type"] = dtype
        self._attrs["has_profiler"] = False

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for reduce ops."""

        input_dims = x._attrs["shape"]
        reduction_axes = self._attrs["reduction_axes"]
        if self._attrs["keepdim"]:
            output_dims = [
                IntImm(1) if idx in set(reduction_axes) else d
                for idx, d in enumerate(input_dims)
            ]
        else:
            # out codegen for reduce ops doesn't rely on the output shape,
            # so it's safe to squeeze the output tensor shape here
            output_dims = [
                d for idx, d in enumerate(input_dims) if idx not in set(reduction_axes)
            ]
        return output_dims

    def _compute_ws_size_strided(
        self, extent, reduction_axis, vector_length, dtype
    ) -> int:
        """
        Compute workspace size for contiguous reduction kernels.
        Reference: cutlass reduction/device/tensor_reduce_affine_strided.h
        """
        k_rank = 4
        k_reduced_rank = 3
        k_inner_rank = k_rank - k_reduced_rank
        k_threads = 256
        target_threadblock_count = 128

        outer_count = 1
        for dim_val in extent[: k_reduced_rank - 1]:
            outer_count *= dim_val
        inner_count = 1
        for dim_val in extent[k_reduced_rank - 1 : k_reduced_rank - 1 + k_inner_rank]:
            inner_count *= dim_val
        extent_c = extent[k_rank - 1]
        vectors_c = (extent_c - 1 + vector_length) // vector_length
        cta_width = k_threads * vector_length

        def _reshape_pow2(ext, count):
            if ext > count:
                return 1
            x = 1
            while count >= ext * 2:
                count >>= 1
                x <<= 1
            return x

        cta_ways = _reshape_pow2(extent_c, cta_width)
        cta_threads_x = k_threads / cta_ways
        threadblock_shape_y = 1
        threadblock_shape_z = min(cta_ways, 64)
        cta_count_x = (vectors_c + cta_threads_x - 1) // cta_threads_x
        cta_count_y = max(1, target_threadblock_count // cta_count_x)
        if cta_count_y * threadblock_shape_y > outer_count:
            cta_count_y = (outer_count + threadblock_shape_y - 1) // threadblock_shape_y
        cta_count_z = max(1, target_threadblock_count / cta_count_y)
        if cta_count_z * threadblock_shape_z > inner_count:
            cta_count_z = (inner_count + threadblock_shape_z - 1) // threadblock_shape_z
        if int(cta_count_z) == 1:
            return 0
        vector_size_bytes = vector_length * get_dtype_size(dtype)
        workspace_stride = extent[k_rank - 1] * vector_size_bytes
        return workspace_stride * outer_count * cta_count_z

    def _compute_workspace_size(
        self, shape: List[IntVar], reduction_axis, dtype
    ) -> int:
        """
        Compute workspace size for the given shape using the same algorithm as
        cutlass's TensorReduction kernel. The only difference is that we use
        the maximum dim value for dynamic dimension, whereas TensorReduction
        uses the real dim value at runtime.
        """
        # Make sure the last dim is static to pre-compute vector_length.
        # Note that this is a temporary constraint. Once we replace TensorReduction
        # with our own col-reduction kernel, we will remove this entire workaround.
        if not shape_utils.is_static_dimension(shape, -1):
            raise NotImplementedError("expect the last dim to be static")
        last_dim = shape[-1].value()
        vector_lens_config = [32, 16, 8, 4, 1]
        vector_length = 1
        for vec_len in vector_lens_config:
            if last_dim % vec_len == 0:
                vector_length = vec_len
                break
        num_dims = 4
        rank = len(shape)
        assert (
            rank <= num_dims
        ), f"expected rank <= num_dims, but got {rank=}, {num_dims=}"
        # adjust reduction axis
        reduction_axis = num_dims - rank + reduction_axis
        all_shape_dims = [
            list(range(d.lower_bound(), d.upper_bound() + 1)) for d in shape
        ]
        max_ws = 0
        # Go through cartesian product of all possible dynamic dim values
        # to find the maximum workspace size. It might be a bit heavy
        # for some cases. However, it's OK for our current use cases where
        # we have a single dynamic axis within a range of several thousand.
        # Moreover, we would remove this entire estimation once we have
        # our own row-reduction kernel.
        for one_dims in itertools.product(*all_shape_dims):
            extent_affine = []
            prefix_dims = [1] * (num_dims - rank)
            # Use dim's upper-bound for computing workspace size
            shape_dims = prefix_dims + list(one_dims)
            # normalize extent_affine list
            # reference: TensorReduction construction in cutlass
            # reduction/device/tensor_reduce.h
            if reduction_axis == 0:
                extent_affine.append(shape_dims[1])
                extent_affine.append(shape_dims[2])
                extent_affine.append(shape_dims[0])
                extent_affine.append(shape_dims[3])
            elif reduction_axis == 1:
                extent_affine.append(shape_dims[0])
                extent_affine.append(shape_dims[2])
                extent_affine.append(shape_dims[1])
                extent_affine.append(shape_dims[3])
            elif reduction_axis == 2:
                extent_affine.append(shape_dims[0])
                extent_affine.append(shape_dims[1])
                extent_affine.append(shape_dims[2])
                extent_affine.append(shape_dims[3])
            else:
                # note that we already ruled out non-col-reduction kernels so that
                # reduction_axis would never be 3. Consequently, we would never
                # invoke contiguous tensor_reduce kernels.
                raise RuntimeError(
                    f"Expected reduction_axis to be within [0, 2], but got {reduction_axis=}"
                )
            max_ws = max(
                max_ws,
                self._compute_ws_size_strided(
                    extent_affine, reduction_axis, vector_length, dtype
                ),
            )
        return max_ws

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        reduction_axes = self._attrs["reduction_axes"]
        if not len(reduction_axes) == 1:
            raise NotImplementedError("Multiple reduction axes are not supported yet")
        input_rank = len(x._attrs["shape"])
        self._attrs["reduction_axes"] = [
            wrap_dim(axis, input_rank) for axis in reduction_axes
        ]
        for axis in self._attrs["reduction_axes"]:
            if axis < 0 or axis >= input_rank:
                raise RuntimeError(
                    "invalid axis {a}, expected in a range [0, {r})".format(
                        a=axis, r=input_rank
                    )
                )
        output_shape = self._infer_shapes(x)
        output_type = self._attrs["output_type"]
        if output_type is None:
            output_type = x._attrs["dtype"]
        output = Tensor(output_shape, src_ops={self}, dtype=output_type)
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        # Under the condition below, we invoke cutlass's TensorReduction kernel,
        # which requires a workspace. The actual size of the required workspace
        # is computed at the runtime before invoking TensorReduction. We tried
        # to simply make the worspace be able to hold the largest tensor using
        # the upperbound of any dynamic dim from the input shape. Unfortunately,
        # this seemed to not work as expected. Particularly, assigning a workspace
        # for a reduce kernel that expects a NULL workspace pointer crashed some
        # of the tests. As a result, we just follow cutlass's algorithm to
        # pre-compute workspace sizes using dims' upperbound values.
        # Note that this is a temprary solution only for col-reduction reduce_sum
        # kernels that invoke cutlass's TensorReduction kernel. Once we have our
        # own implementation, we will remove the workaround.
        if self._attrs["op"] == "reduce_sum" and (
            self._attrs["reduction_axes"][0] != input_rank - 1
        ):
            ws_size = self._compute_workspace_size(
                x._attrs["shape"], self._attrs["reduction_axes"][0], x.dtype()
            )
            _LOGGER.info(f'allocating {ws_size} for tensor {x._attrs["name"]}')
            self._attrs["workspace"] = ws_size
        return output

    def _get_op_attributes(self):
        return {
            "dim": self._attrs["reduction_axes"],
            "dtype": self._attrs["output_type"],
            "keepdim": self._attrs["keepdim"],
        }

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
