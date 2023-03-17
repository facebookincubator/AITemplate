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
Batched nms.
"""
import itertools
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import (  # noqa
    _create_host_zero_tensor,
    IntImm,
    Operator,
    Tensor,
)
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
M == {{x_dim0}} && K == {{x_dim1}}
"""
)


class batched_nms(Operator):
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU) in a batched fashion.

    NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box.

    Note: if multiple boxes have the exact same score and satisfy the IoU criterion with respect to a reference box, the selected box is not guaranteed to be the same for different backends.

     * :attr:`iouThreshold` identifies the intersection-over-union (IoU) threshold which is used to discards all overlapping boxes with IoU > iouThreshold. By default 0.5.

     * :attr:`keep_n` identifies the number of boxes to return, by default -1 to return all.

    Args:
        boxes (Tensor[N, 4])), boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``), and have been sorted in decreasing order of scores.

    Returns:
        Tensor: "keep" (Tensor[N]) in which each element indicates if the corresponding box is removed (element=0) or not (element=1).
    """

    def __init__(self, iou_threshold=0.5, keep_n=-1) -> None:
        """Op Initialization"""
        super().__init__()
        self._attrs["op"] = "batched_nms"
        self._attrs["has_profiler"] = False
        self._attrs["keep_n"] = keep_n
        self._attrs["iou_threshold"] = iou_threshold
        self.exec_key_template = EXEC_KEY_TEMPLATE

    def _infer_shape(self, x: List[int]):
        """infer output shape"""
        return [x[0]]

    def _infer_shapes(self, x: Tensor):
        """infer output shape"""
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = []
        for idx in range(len(y_shapes[0])):
            output_shape.append(
                shape_utils.gen_int_var(values=unique([d[idx] for d in y_shapes]))
            )
        return output_shape

    def __call__(self, x: Tensor) -> Tensor:
        """call the function"""
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype="int64")
        boxes_num = x._attrs["shape"][0]._attrs["values"][0]
        col_blocks = int((boxes_num + 64 - 1) / 64)
        tmp_space = col_blocks * boxes_num
        tmp_c = _create_host_zero_tensor(
            [IntImm(tmp_space)], dst_ops={self}, dtype="int64"
        )
        self._attrs["inputs"].append(tmp_c)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "iou_threshold": self._attrs["iou_threshold"],
            "keep_n": self._attrs["keep_n"],
        }

    def gen_function(self) -> str:
        """call backend function"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
