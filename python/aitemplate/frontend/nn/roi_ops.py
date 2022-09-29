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
RoiAlign-family modules.
"""
from ...compiler.ops import multi_level_roi_align, roi_align
from .module import Module


class RoiAlign(Module):
    def __init__(
        self,
        num_rois,
        pooled_size,
        sampling_ratio,
        spatial_scale,
        position_sensitive,
        continuous_coordinate,
    ):
        super().__init__()
        self.op = roi_align(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
        )

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        rois = args[1]
        return self.op(x, rois)


class FPNRoiAlign(Module):
    def __init__(
        self,
        num_rois,
        pooled_size,
        sampling_ratio,
        spatial_scale,
        position_sensitive,
        continuous_coordinate,
        im_shape,
    ):
        super().__init__()
        self.op = multi_level_roi_align(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
            im_shape,
        )

    def forward(self, *args):
        assert len(args) >= 2
        x = args[0]
        rois = args[1]
        return self.op(x, rois)
