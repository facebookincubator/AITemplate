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
from aitemplate.frontend import nn, Tensor


class fast_rcnn_inference:
    def __init__(
        self,
        im_shape: Tuple[int, int],
        num_rois: int,
        num_classes: int,
        clip_box: bool = True,
        nms_on: bool = True,
        use_topk: bool = True,
        topk_per_image: int = 100,
        iou_thresh: float = 0.5,
        roi_align_on: bool = False,
        batch_size: int = 1,
        dtype: str = "float16",
    ):
        self.im_h, self.im_w = im_shape
        self.num_rois = num_rois
        self.num_classes = num_classes
        self.dtype = dtype
        self.clip_box = clip_box
        self.topk_per_image = topk_per_image
        self.iou_threshold = iou_thresh
        self.nms_on = nms_on
        self.use_topk = use_topk
        self.roi_align_on = roi_align_on
        self.batch_size = batch_size
        self.class_agnostic_nms = False

    def __call__(self, boxes, scores, deltas):
        """
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.

            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.

            deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box
                transform (see :class:`fast_rcnn_inference.box_transform`).

        Returns:
            proposals.
        """
        proposals = self.box_transform(boxes, deltas)
        if self.nms_on:
            return self.nms_wrapper(proposals, scores)
        else:
            return proposals

    def nms_wrapper(self, proposals, scores):
        N = self.batch_size
        proposals_p = ops.permute102()(proposals)
        scores_x = ops.dynamic_slice()(
            scores, start_indices=[0, 0], end_indices=[self.num_rois, self.num_classes]
        )

        OP = ops.efficient_nms(
            self.num_rois // N, self.topk_per_image, self.iou_threshold, 0
        )
        args = (
            ops.reshape()(proposals_p, [N, -1, self.num_classes, 4]),
            ops.reshape()(scores_x, [N, -1, self.num_classes]),
        )
        detections = OP(*args)
        if self.roi_align_on:
            batch_inds = Tensor(
                shape=[N, self.topk_per_image, 1],
                dtype=self.dtype,
                name="batch_inds_mask",
                value=0,
            )
            rois = ops.reshape()(
                ops.concatenate()([batch_inds, detections[1]], dim=2), [-1, 5]
            )
            return detections + (rois,)
        else:
            return detections

    def layout_transform(self, delta):
        return ops.permute210()(
            ops.reshape()(delta, [1, self.num_rois, self.num_classes])
        )

    def apply_weight(self, deltas, weights=(0.1, 0.2)):
        ww = weights[0]
        wh = weights[1]

        deltas_r = ops.reshape()(deltas, [self.num_rois, -1, 4])
        (delta_x, delta_y, delta_w, delta_h) = ops.split()(deltas_r, 1, dim=2)
        delta_xm = delta_x * ww
        delta_ym = delta_y * ww
        delta_wm = delta_w * wh
        delta_hm = delta_h * wh

        return (
            self.layout_transform(delta_xm),
            self.layout_transform(delta_ym),
            self.layout_transform(delta_wm),
            self.layout_transform(delta_hm),
        )

    def box_transform(self, boxes, deltas):
        """
        The box-to-box transform defined in R-CNN. The transformation is parameterized by 4 deltas: (dx, dy, dw, dh). The transformation scales the box’s width and height by exp(dw), exp(dh) and shifts a box’s center by the offset (dx * width, dy * height).
        """
        const_0_5 = 0.5

        (delta_x, delta_y, delta_w, delta_h) = self.apply_weight(deltas)

        boxes_r = ops.reshape()(boxes, [self.num_rois, 4])
        (anchor_x1, anchor_y1, anchor_x2, anchor_y2) = ops.split()(boxes_r, 1, dim=1)
        widths = ops.reshape()(anchor_x2 - anchor_x1, [self.num_rois, 1])
        heights = ops.reshape()(anchor_y2 - anchor_y1, [self.num_rois, 1])

        width_mid = widths * const_0_5
        height_mid = heights * const_0_5
        ctr_x = anchor_x1 + width_mid
        ctr_y = anchor_y1 + height_mid

        pred_ctr_x = (delta_x * widths) + ctr_x

        pred_ctr_y = (delta_y * heights) + ctr_y
        pred_w = ops.exp(delta_w) * widths
        pred_h = ops.exp(delta_h) * heights

        p_x1 = pred_ctr_x - (const_0_5 * pred_w)
        p_y1 = pred_ctr_y - (const_0_5 * pred_h)
        p_x2 = pred_ctr_x + (const_0_5 * pred_w)
        p_y2 = pred_ctr_y + (const_0_5 * pred_h)

        if self.clip_box:
            f_x1, f_y1, f_x2, f_y2 = self.box_clip(p_x1, p_y1, p_x2, p_y2)
            proposals = ops.concatenate()([f_x1, f_y1, f_x2, f_y2], dim=2)
        else:
            proposals = ops.concatenate()([p_x1, p_y1, p_x2, p_y2], dim=2)

        return proposals

    def box_clip(self, p_x1, p_y1, p_x2, p_y2):
        x_min = 0
        x_max_h = self.im_h
        x_max_w = self.im_w

        f_x1 = ops.hardtanh(p_x1, x_min, x_max_w)
        f_y1 = ops.hardtanh(p_y1, x_min, x_max_h)
        f_x2 = ops.hardtanh(p_x2, x_min, x_max_w)
        f_y2 = ops.hardtanh(p_y2, x_min, x_max_h)
        return f_x1, f_y1, f_x2, f_y2


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores

    , and a postprocess procedure.
    """

    def __init__(self, cfg, im_shape):
        super().__init__()
        in_channel = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.cls_score = nn.Linear(in_channel, num_classes + 1)
        self.bbox_pred = nn.Linear(in_channel, num_classes * 4)

        self.postprocess = fast_rcnn_inference(
            im_shape=im_shape,
            num_classes=num_classes,
            num_rois=cfg.MODEL.RPN.POST_NMS_TOPK_TEST * cfg.SOLVER.IMS_PER_BATCH,
            use_topk=cfg.POSTPROCESS.USE_TOPK,
            roi_align_on=True if cfg.MODEL.MASK_ON else False,
            topk_per_image=cfg.POSTPROCESS.TOPK,
            iou_thresh=cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0],
            clip_box=True,
            nms_on=True,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
        )

    def forward(self, x, proposals):
        rcnn_logit = self.cls_score(x)
        rcnn_logit = ops.softmax()(rcnn_logit, -1)
        rcnn_reg = self.bbox_pred(x)
        return self.postprocess(proposals, rcnn_logit, rcnn_reg)
