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
Efficient nms.
"""
import itertools
import logging
import os
import re
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223


_LOGGER = logging.getLogger(__name__)

# TODO: change to column last
SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}BS = {{x_dim0}};
{{indent}}{{dtype}}NB = {{x_dim1}};
{{indent}}{{dtype}}NC = {{x_dim2}};
{{indent}}{{dtype}}SZ = {{x_dim3}};
{{indent}}{{dtype}}NO = BS;
{{indent}}{{dtype}}CO = {{nmsMaxOut}};
{{indent}}{{dtype}}HO = SZ;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
num_batch == {{x_dim0}} && num_rois == {{x_dim1}} && num_classes == {{x_dim2}}
"""
)


class efficient_nms(Operator):
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box.

    Note: if multiple boxes have the exact same score and satisfy the IoU criterion with respect to a reference box, the selected box is not guaranteed to be the same for different backends.

     * :attr:`preNmsTop` identifies the maximum number of boxes to take.

     * :attr:`nmsMaxOut` identifies the maximum number of boxes to reserve after the operation.

     * :attr:`iouThreshold` identifies the intersection-over-union (IoU) threshold which is used to discards all overlapping boxes with IoU > iouThreshold.

     * :attr:`minBoxSize` identifies the minimum box size, if a box has size less than this value, it will be removed before the non-maximum suppression.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """

    def __init__(
        self, preNmsTop=2000, nmsMaxOut=200, iouThreshold=0.5, minBoxSize=0
    ) -> None:
        """Initializes efficient_nms"""
        super().__init__()
        self._attrs["op"] = "efficient_nms"
        self._attrs["preNmsTop"] = preNmsTop
        self._attrs["nmsMaxOut"] = nmsMaxOut
        self._attrs["iouThreshold"] = iouThreshold
        self._attrs["minBoxSize"] = minBoxSize
        self._attrs["has_profiler"] = True
        self._attrs["workspace"] = 0
        self.exec_key_template = EXEC_KEY_TEMPLATE
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]):
        """infer the output shape for nms op"""
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            nmsMaxOut=self._attrs["nmsMaxOut"],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204  # noqa: P204
        return [int(output["NO"]), int(output["CO"]), int(output["HO"])]

    def _infer_shapes(self, x: Tensor, w: Tensor):
        """infer the output shape for nms op"""
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["KH"] = w_shape[0]
        self._attrs["KW"] = w_shape[1]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
        ]
        return output_shape

    def __call__(self, boxes: Tensor, scores: Tensor) -> Tensor:
        """Performs shape inference and returns an output tensor."""
        self._attrs["inputs"] = [boxes, scores]
        self._set_depth()
        self._extract_exec_path(boxes)
        output_shape = self._infer_shapes(boxes, scores)

        x = boxes
        num_detections = Tensor(
            [output_shape[0], IntImm(1)], dtype="int64", src_ops={self}
        )
        detection_boxes = Tensor(
            output_shape,
            src_ops={self},
            dtype=x._attrs["dtype"],
        )
        detection_scores = Tensor(
            output_shape[:-1],
            src_ops={self},
            dtype=x._attrs["dtype"],
        )
        detection_classes = Tensor(output_shape[:-1], dtype="int64", src_ops={self})
        output = (num_detections, detection_boxes, detection_scores, detection_classes)
        self._attrs["outputs"] = [
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
        ]
        return output

    def _get_op_attributes(self):
        return {
            "iouThreshold": self._attrs["iouThreshold"],
            "minBoxSize": self._attrs["minBoxSize"],
            "nmsMaxOut": self._attrs["nmsMaxOut"],
            "preNmsTop": self._attrs["preNmsTop"],
        }

    def _gen_exec_key(self, shape):
        """rendering shape info"""
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1] * shape[2],
            x_dim2=shape[2],
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def gen_function(self) -> str:
        """call backend functions"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_profiler".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, workdir)

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape):
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        cmd.append(x_shape[0])
        cmd.append(x_shape[1] * x_shape[2])
        cmd.append(x_shape[2])
        command = [str(x) for x in cmd]
        _LOGGER.info("profiling cmd: {}".format(command))
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices):
        runner = backend.profiler_runner.Runner(devices, self._attrs["name"])
        cfg = self._attrs["op"]
        x_shape = self._invert_exec_key(exec_key)
        command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape)
        runner.push(cfg, command)
        runner.join()
        result = runner.pull()

        if len(result) == 0:
            raise RuntimeError(
                "Profile workload: " f"{exec_key}" " failed. " f"Results: {result}."
            )

        out = min(result, key=lambda x: x[1].duration)
        workspace = out[1].workspace
        return workspace

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=None,
    ):
        """Profile to compute the NMS Op workspace size."""
        if devices is None:
            devices = [0]

        workloads = list(self._attrs["exec_path"].keys())
        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])

        for wkl in workloads:
            _LOGGER.info(
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            workspace = self._profile_single_workload(profiler_prefix, wkl, devices)
            self._attrs["workspace"] = max(self._attrs["workspace"], workspace)
