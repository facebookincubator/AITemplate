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
Roi.
"""
import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,R1732,W0613
logging.basicConfig(level=logging.INFO)

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}KH = {{pooled_size}};
{{indent}}{{dtype}}KW = {{pooled_size}};
{{indent}}{{dtype}}NO = {{num_rois}};
{{indent}}{{dtype}}CO = CI;
{{indent}}{{dtype}}HO = {{pooled_size}};
{{indent}}{{dtype}}WO = {{pooled_size}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class roi_ops_base(Operator):
    """
    Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

     * :attr:`num_rois` identifies the number of RoIs in the input.

     * :attr:`pooled_size` identifies the size of the pooling section, i.e., the size of the output (in bins or pixels) after the pooling
       is performed, as (height, width).

     * :attr:`sampling_ratio` is the number of sampling points in the interpolation grid
       used to compute the output value of each pooled output bin. If > 0,
       then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
       <= 0, then an adaptive number of grid points are used (computed as
       ``ceil(roi_width / output_width)``, and likewise for height).

     * :attr:`spatial_scale` is a scaling factor that maps the box coordinates to
       the input coordinates. For example, if your boxes are defined on the scale
       of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
       the original image), you'll want to set this to 0.5.

     * :attr:`position_sensitive`, a bool value.

     * :attr:`continuous_coordinate`. a bool value.

    Args:
        x (Tensor[N, H, W, C]): the feature map, i.e. a batch with ``N`` elements. Each element contains ``C`` feature maps of dimensions ``H x W``.
        rois (Tensor[roi_batch, 5]): the list of RoIs and each ROI contains the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``, and the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Return:
        Tensor[roi_batch, pooled_size, pooled_size, C]: the fixed-size feature maps, i.e., the pooled RoIs.

    """

    def __init__(
        self,
        num_rois,
        pooled_size,
        sampling_ratio,
        spatial_scale,
        position_sensitive,
        continuous_coordinate,
    ) -> None:
        super().__init__()
        self._attrs["op"] = "roi_align"
        self._attrs["num_rois"] = num_rois
        self._attrs["sampling_ratio"] = sampling_ratio
        self._attrs["spatial_scale"] = spatial_scale
        self._attrs["position_sensitive"] = position_sensitive
        self._attrs["continuous_coordinate"] = continuous_coordinate
        self._attrs["pooled_size"] = pooled_size
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            num_rois=self._attrs["num_rois"],
            pooled_size=self._attrs["pooled_size"],
            position_sensitive=self._attrs["position_sensitive"],
        )

        output = {}
        exec(eval_func, output)  # noqa: P204  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape):
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2], x_dim3=shape[3]
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self):
        signature = "roi_align: num_rois=[{num_rois}], \
                                sampling_ratio=[{sampling_ratio}], \
                                spatial_scale=[{spatial_scale}], \
                                position_sensitive=[{position_sensitive}], \
                                continuous_coordinate=[{continuous_coordinate}], \
                                pooled_size=[{pooled_size}]".format(
            num_rois=self._attrs["num_rois"],
            sampling_ratio=self._attrs["sampling_ratio"],
            spatial_scale=self._attrs["spatial_scale"],
            position_sensitive=self._attrs["position_sensitive"],
            continuous_coordinate=self._attrs["continuous_coordinate"],
            pooled_size=self._attrs["pooled_size"],
        )
        return signature

    def __call__(self, x: Tensor, rois: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x, rois]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        target_attrs = [
            "continuous_coordinate",
            "num_rois",
            "pooled_size",
            "position_sensitive",
            "sampling_ratio",
            "spatial_scale",
        ]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
