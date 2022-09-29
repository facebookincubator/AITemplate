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


class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (or `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    def __init__(
        self,
        num_rois: int,
        num_classes: int,
        feat_dim: int,
        conv_dim: int,
        pooled_size: int,
        im_shape: Tuple[int, int],
    ):
        super().__init__()
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
        in_channel = feat_dim
        mid_channel = conv_dim

        self.mask_fcn1 = nn.Conv2dBiasRelu(in_channel, mid_channel, 3, 1, 1)
        self.mask_fcn2 = nn.Conv2dBiasRelu(mid_channel, mid_channel, 3, 1, 1)
        self.mask_fcn3 = nn.Conv2dBiasRelu(mid_channel, mid_channel, 3, 1, 1)
        self.mask_fcn4 = nn.Conv2dBiasRelu(mid_channel, mid_channel, 3, 1, 1)
        self.deconv = nn.ConvTranspose2dBiasRelu(mid_channel, mid_channel, 2, 2, 0)
        self.predictor = nn.Conv2dBiasSigmoid(mid_channel, num_classes, 1, 1, 0)

    def forward(self, feat, rois):
        roi_feat = self.roi_align(feat[0], feat[1], feat[2], feat[3], rois)
        conv1 = self.mask_fcn1(roi_feat)
        conv2 = self.mask_fcn2(conv1)
        conv3 = self.mask_fcn3(conv2)
        conv4 = self.mask_fcn4(conv3)
        upsp = self.deconv(conv4)
        mask = self.predictor(upsp)
        return mask
