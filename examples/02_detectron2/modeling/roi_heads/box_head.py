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
from typing import Tuple

from aitemplate.compiler import ops
from aitemplate.frontend import nn

from .fast_rcnn import FastRCNNOutputLayers


class FastRCNNConvFCHead(nn.Module):
    """
    A head with a multi_level roi align layer and two fc layers.
    """

    def __init__(
        self,
        num_rois: int,
        num_classes: int,
        feat_dim: int,
        fc_dim: int,
        pooled_size: int,
        im_shape: Tuple[int, int],
    ):
        super().__init__()
        self.num_rois = num_rois
        HH, WW = im_shape
        self.roi_align = ops.multi_level_roi_align(
            num_rois=num_rois,
            pooled_size=pooled_size,
            spatial_scale=1.0,
            sampling_ratio=0,
            position_sensitive=False,
            continuous_coordinate=False,
            im_shape=im_shape,
        )
        in_channel = int(feat_dim * pooled_size**2)
        mid_channel = fc_dim

        self.fc1 = nn.Linear(in_channel, mid_channel, specialization="relu")
        self.fc2 = nn.Linear(mid_channel, mid_channel, specialization="relu")

    def forward(self, feat, rois):
        roi_feat = self.roi_align(feat[0], feat[1], feat[2], feat[3], rois)
        roi_feat = ops.reshape()(roi_feat, [ops.size()(roi_feat, 0), -1])
        fc1 = self.fc1(roi_feat)
        fc2 = self.fc2(fc1)
        return fc2


def build_box_head(cfg, input_shape):
    """
    Build a box head through `FastRCNNOutputLayers`.
    """
    return FastRCNNOutputLayers(cfg, input_shape)
