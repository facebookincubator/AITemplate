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
Multi level roi_align.
"""

from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.vision_ops.roi_ops.roi_ops import roi_ops_base

# pylint: disable=C0103


class multi_level_roi_align(roi_ops_base):
    """
    Performs Multiple level Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

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

     * :attr:`continuous_coordinate`, a bool value.

     * :attr:`im_shape`, original image shape.

    Args:
        p1 (Tensor[N, H//4, W//4, C]): the feature map, i.e. a batch with ``N`` elements. Each element contains ``C`` feature maps of dimensions ``(H//4) x (W//4)``.
        p2 (Tensor[N, H//8, W//8, C]): the feature map, i.e. a batch with ``N`` elements. Each element contains ``C`` feature maps of dimensions ``(H//8) x (W//8)``.
        p3 (Tensor[N, H//16, W//16, C]): the feature map, i.e. a batch with ``N`` elements. Each element contains ``C`` feature maps of dimensions ``(H//16) x (W//16)``.
        p4 (Tensor[N, H//32, W//32, C]): the feature map, i.e. a batch with ``N`` elements. Each element contains ``C`` feature maps of dimensions ``(H//32) x (W//32)``.
        rois (Tensor[roi_batch, 5]): the list of RoIs and each ROI contains the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``, and the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Return:
        Tensor[num_rois * N, pooled_size, pooled_size, C]: the fixed-size feature maps, i.e., the pooled RoIs.

    """

    def __init__(
        self,
        num_rois,
        pooled_size,
        sampling_ratio,
        spatial_scale,
        position_sensitive,
        continuous_coordinate,
        im_shape,
    ) -> None:
        super().__init__(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
        )
        self._attrs["op"] = "multi_level_roi_align"
        self._attrs["im_shape"] = im_shape

    def _infer_shape(self, x: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            num_rois=self._attrs["num_rois"] * x[0],
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

    def __call__(
        self, p2: Tensor, p3: Tensor, p4: Tensor, p5: Tensor, rois: Tensor
    ) -> List[Tensor]:
        self._attrs["inputs"] = [p2, p3, p4, p5, rois]
        x = p2
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        attr = super()._get_op_attributes()
        attr["im_shape"] = self._attrs["im_shape"]

        return attr
