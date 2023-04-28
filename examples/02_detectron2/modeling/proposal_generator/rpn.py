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
import numpy as np
import torch
from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from aitemplate.frontend import nn


class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(
        self,
        in_planes,
        rpn_dim=256,
        scales=((32,), (64,), (128,), (256,), (512,)),
        ratios=(0.5, 1, 2),
    ):
        super().__init__()
        num_anchors = len(scales) * len(ratios)
        self.conv = nn.Conv2dBiasRelu(in_planes, rpn_dim, 3, 1, 1)
        self.objectness_logits = nn.Conv2dBiasSigmoid(rpn_dim, num_anchors, 1, 1, 0)
        self.anchor_deltas = nn.Conv2dBias(rpn_dim, num_anchors * 4, 1, 1, 0)

    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for _, x in features.items():
            t = ops.conv2d_bias_relu(stride=1, pad=1)(
                x, self.conv.weight.tensor(), self.conv.bias.tensor()
            )
            logit = ops.conv2d_bias_sigmoid(stride=1, pad=0)(
                t,
                self.objectness_logits.weight.tensor(),
                self.objectness_logits.bias.tensor(),
            )
            reg = ops.conv2d_bias(stride=1, pad=0)(
                t, self.anchor_deltas.weight.tensor(), self.anchor_deltas.bias.tensor()
            )
            pred_objectness_logits.append(logit)
            pred_anchor_deltas.append(reg)

        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def __init__(
        self,
        cfg,
        im_shape,
        dtype="float16",
    ):
        super().__init__()
        # fmt: off
        in_planes          = cfg.MODEL.FPN.OUT_CHANNELS
        batch_size         = cfg.SOLVER.IMS_PER_BATCH
        rpn_pre_nms_top_n  = cfg.MODEL.RPN.PRE_NMS_TOPK_TEST
        rpn_post_nms_top_n = cfg.MODEL.RPN.POST_NMS_TOPK_TEST
        self.iou_threshold = cfg.MODEL.RPN.NMS_THRESH
        self.rpn_min_size  = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.scales        = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.ratios        = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        # fmt: on
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.topk = rpn_pre_nms_top_n
        self.dtype = dtype
        self.im_shape = im_shape
        self.feat_strides = (4, 8, 16, 32, 64)
        self.batch_size = batch_size
        self.batch_inds = np.zeros((batch_size, rpn_post_nms_top_n, 1)).astype(dtype)

        self.rpn_head = StandardRPNHead(
            in_planes,
            in_planes,
            scales=self.scales[0],
            ratios=self.ratios,
        )

        self.proposal = nn.FPNProposal(
            im_shape=im_shape,
            feat_strides=self.feat_strides,
            scales=self.scales,
            ratios=self.ratios,
            clip_box=True,
            nms_on=False,
            rpn_pre_nms_top_n=self.rpn_pre_nms_top_n,
            rpn_post_nms_top_n=self.rpn_post_nms_top_n,
            iou_threshold=self.iou_threshold,
            rpn_min_size=self.rpn_min_size,
            batch_size=batch_size,
        )

    def forward(self, features):
        N = self.batch_size
        pred_logits, pred_deltas = self.rpn_head(features)
        pred_rois = self.proposal(pred_deltas)

        proposal_list = []
        score_list = []
        for rois, logit in zip(pred_rois, pred_logits):
            rois = ops.reshape()(rois, [N, -1, 4])
            if self.topk > 0 and rois.shape()[1].value() > self.topk:
                _, score_inds = ops.topk(k=self.topk)(ops.reshape()(logit, [N, -1]))
                boxes_topk = ops.batch_gather()(rois, score_inds)
                scores_topk = ops.batch_gather()(
                    ops.reshape()(logit, [N, -1, 1]), score_inds
                )
                proposal_list.append(boxes_topk)
                score_list.append(ops.reshape()(scores_topk, [N, -1]))
            else:
                proposal_list.append(rois)
                score_list.append(ops.reshape()(logit, [N, -1]))

        proposals_concat = ops.concatenate()(proposal_list, dim=1)
        scores_concat = ops.concatenate()(score_list, dim=1)
        scores_r = ops.reshape()(scores_concat, [N, -1])
        proposals_r = ops.reshape()(proposals_concat, [N, -1, 4])

        dets = ops.nms(
            self.rpn_pre_nms_top_n,
            self.rpn_post_nms_top_n,
            self.iou_threshold,
            self.rpn_min_size,
        )(proposals_r, scores_r)

        batch_inds = Tensor(
            shape=[N, self.rpn_post_nms_top_n, 1],
            dtype=self.dtype,
            name="batch_inds",
            value=0,
        )
        ret = ops.reshape()(ops.concatenate()([batch_inds, dets], dim=2), [-1, 5])
        return ret, ops.reshape()(dets, [-1, 4])

    def set_anchors(self, mod):
        param = {"batch_inds": self.batch_inds.copy()}
        for idx, _ in enumerate(self.feat_strides):
            param["anchors_%d" % (idx + 2)] = self.proposal._anchors[idx].copy()

        weights = {name: torch.from_numpy(w).cuda().half() for name, w in param.items()}
        for name, weight in weights.items():
            mod.set_constant_with_tensor(name, weight)

    def get_params(self):
        params = {
            "anchors_%d" % (idx + 2): anchor.copy()
            for idx, anchor in enumerate(self.proposal._anchors)
        }
        params["batch_inds"] = self.batch_inds
        return params


def build_rpn_head(cfg, input_shape):
    return RPN(cfg, input_shape)
