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
FPNProposal module.
"""
import numpy as np

from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from aitemplate.frontend.nn.proposal import generate_shifted_anchors, Proposal


def generate_fpn_anchors(im_h, im_w, feat_strides, scales, ratios, batch_size, dtype):
    """Enumerate anchors for all levels"""
    anchors = []
    for feat_stride, scale in zip(feat_strides, scales):
        anchors.append(
            generate_shifted_anchors(
                im_h,
                im_w,
                feat_stride,
                np.array(scale, dtype="float32"),
                np.array(ratios, dtype="float32"),
                batch_size,
                dtype,
            )
        )
    return anchors


class FPNProposal(Proposal):
    def __init__(
        self,
        im_shape,
        feat_strides=(4, 8, 16, 32, 64),
        scales=((32,), (64,), (128,), (256,), (512,)),
        ratios=(0.5, 1, 2),
        clip_box=True,
        nms_on=True,
        rpn_pre_nms_top_n=6000,
        rpn_post_nms_top_n=300,
        iou_threshold=0.3,
        rpn_min_size=0,
        level=-1,
        batch_size=1,
        dtype="float16",
    ):
        super().__init__(
            im_shape,
            feat_strides,
            scales,
            ratios,
            clip_box,
            nms_on,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            iou_threshold,
            rpn_min_size,
            level,
            generate_fpn_anchors,
            batch_size,
            dtype,
        )

    def forward(self, *args):
        assert len(args) >= 1

        reg_list = args[0]
        bbox_deltas_list = []
        anchors_list = []
        split_size_or_sections = []
        for idx, x_bbox_deltas in enumerate(reg_list):
            bbox_deltas = ops.reshape()(x_bbox_deltas, [-1, 4])
            anchors = Tensor(
                shape=self._anchors[idx].shape, name="anchors_%d" % (idx + 2)
            )
            bbox_deltas_list.append(bbox_deltas)
            anchors_list.append(anchors)
            split_size_or_sections.append(self._anchors[idx].shape[0])

        bbox_deltas_cat = ops.concatenate()(bbox_deltas_list, dim=0)
        anchors_cat = ops.concatenate()(anchors_list, dim=0)
        proposals = self.box_transform(bbox_deltas_cat, anchors_cat)

        proposals = ops.split()(proposals, split_size_or_sections, dim=0)

        if self.nms_on:
            scores = args[1]
            scores_r = ops.reshape()(scores, [1, -1])
            proposals_r = ops.reshape()(proposals, [1, -1, 4])
            dets = ops.nms(
                self.rpn_pre_nms_top_n,
                self.rpn_post_nms_top_n,
                self.iou_threshold,
                self.rpn_min_size,
            )(proposals_r, scores_r)
            # prepare for roi-align for mask head
            batch_inds = Tensor(
                shape=[1, self.rpn_post_nms_top_n, 1],
                dtype=self.dtype,
                name="batch_inds",
                value=0,
            )
            ret = ops.reshape()(ops.concatenate()([batch_inds, dets], dim=2), [-1, 5])
            return ret, dets
        return list(proposals)
