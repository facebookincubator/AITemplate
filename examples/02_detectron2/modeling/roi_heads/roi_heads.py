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
from typing import Dict

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor

from .box_head import build_box_head, FastRCNNConvFCHead
from .mask_head import MaskRCNNConvUpsampleHead


class StandardROIHeads(nn.Module):
    """
    The StandardROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.mask_on = cfg.MODEL.MASK_ON
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.box_predictor = build_box_head(cfg, input_shape)

        self.box_head = FastRCNNConvFCHead(
            num_rois=cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            feat_dim=cfg.MODEL.FPN.OUT_CHANNELS,
            fc_dim=cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            pooled_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            im_shape=input_shape,
        )
        if cfg.MODEL.MASK_ON:
            self.mask_head = MaskRCNNConvUpsampleHead(
                num_rois=cfg.POSTPROCESS.TOPK,
                num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
                feat_dim=cfg.MODEL.FPN.OUT_CHANNELS,
                conv_dim=cfg.MODEL.ROI_MASK_HEAD.CONV_DIM,
                pooled_size=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
                im_shape=input_shape,
            )

    def get_shape(self, x):
        shape = [it.value() for it in x._attrs["shape"]]
        return shape

    def forward(self, features: Dict[str, Tensor], rois: Tensor, proposals: Tensor):

        box_features = [features[f] for f in self.in_features]
        roi_feat = self.box_head(box_features, rois)
        detections = self.box_predictor(roi_feat, proposals)
        if self.mask_on:
            num_dets, boxes, probs, class_pred, mask_rois = detections
            pred_mask_logits = self.mask_head(box_features, mask_rois)

            num_rois, roi_size, _, num_classes = self.get_shape(pred_mask_logits)
            batch_size = self.get_shape(boxes)[0]
            batch_rois = num_rois // batch_size

            pred_mask_logits = ops.permute021()(
                ops.reshape()(pred_mask_logits, [num_rois, -1, num_classes])
            )
            indices = ops.reshape()(class_pred, [num_rois, 1])
            mask_probs_pred = ops.batch_gather()(pred_mask_logits, indices)
            mask_probs_pred = ops.reshape()(
                mask_probs_pred, [batch_size, batch_rois, roi_size, roi_size]
            )
            return num_dets, boxes, probs, class_pred, mask_probs_pred
        else:
            return detections


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads through `StandardROIHeads`.
    """
    return StandardROIHeads(cfg, input_shape)
