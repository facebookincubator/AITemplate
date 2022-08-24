# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import itertools
from typing import List

import jinja2

from ..... import backend
from .....backend import registry
from .....utils import shape_utils
from ....base import Operator, Tensor  # noqa

# pylint: disable=C0103,W0221,W0102,W0223

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
M == {{x_dim0}} && K == {{x_dim1}}
"""
)


class batched_nms(Operator):
    """batched_nms implementation

    Parameters
    ----------
    Operator : [type]
        [description]
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
        tmp_c = Tensor([tmp_space], dst_ops={self}, dtype="int64")
        self._attrs["inputs"].append(tmp_c)
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """call backend function"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
