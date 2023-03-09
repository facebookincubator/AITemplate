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
Proposal module.
"""
import math

import numpy as np

from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend.nn.module import Module


def _mkanchors(widths, heights, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    widths = widths[:, np.newaxis]
    heights = heights[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (widths - 1),
            y_ctr - 0.5 * (heights - 1),
            x_ctr + 0.5 * (widths - 1),
            y_ctr + 0.5 * (heights - 1),
        )
    )
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    width = anchor[2] - anchor[0] + 1
    height = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (width - 1)
    y_ctr = anchor[1] + 0.5 * (height - 1)
    return width, height, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    width, height, x_ctr, y_ctr = _whctrs(anchor)
    size = width * height
    size_ratios = size / ratios
    widths = np.round(np.sqrt(size_ratios))
    heights = np.round(widths * ratios)
    anchors = _mkanchors(widths, heights, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    width, height, x_ctr, y_ctr = _whctrs(anchor)
    widths = width * scales
    heights = height * scales
    anchors = _mkanchors(widths, heights, x_ctr, y_ctr)
    return anchors


def generate_anchors(ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    anchors = []
    for size in scales:
        area = size**2.0
        for aspect_ratio in ratios:
            im_w = math.sqrt(area / aspect_ratio)
            im_h = aspect_ratio * im_w
            im_x0, im_y0, im_x1, im_y1 = (
                -im_w / 2.0,
                -im_h / 2.0,
                im_w / 2.0,
                im_h / 2.0,
            )
            anchors.append([im_x0, im_y0, im_x1, im_y1])
    return np.array(anchors)


def generate_shifted_anchors(
    im_h,
    im_w,
    feat_stride,
    scales,
    ratios,
    batch_size,
    dtype,
):
    """
    Enumerate all shifted anchors
    """
    height, width = np.ceil(im_h / feat_stride), np.ceil(im_w / feat_stride)

    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    anchors = generate_anchors(scales=scales, ratios=ratios)
    num_anchors = anchors.shape[0]
    num_shifts = shifts.shape[0]
    anchors = anchors.reshape((1, num_anchors, 4)) + shifts.reshape(
        (1, num_shifts, 4)
    ).transpose((1, 0, 2))
    anchors = anchors.reshape((1, num_shifts * num_anchors, 4))
    exp_anchors = np.repeat(anchors, repeats=batch_size, axis=0).reshape(-1, 4)
    return exp_anchors.astype(dtype)


def gen_batch_inds(
    batch_size,
    rpn_post_nms_top_n,
    dtype="float16",
):
    if batch_size > 1:
        inds = np.arange(batch_size)
        batch_inds = np.repeat(inds.reshape(-1, 1), repeats=rpn_post_nms_top_n, axis=1)
        return batch_inds.reshape(batch_size, rpn_post_nms_top_n, 1).astype(dtype)
    else:
        return np.zeros((batch_size, rpn_post_nms_top_n, 1)).astype(dtype)


class Proposal(Module):
    def __init__(
        self,
        im_shape,
        feat_stride=16,
        scales=(32, 64, 128, 256, 512),
        ratios=(0.5, 1, 2),
        clip_box=True,
        nms_on=True,
        rpn_pre_nms_top_n=6000,
        rpn_post_nms_top_n=300,
        iou_threshold=0.3,
        rpn_min_size=0,
        level=-1,
        f_proc=None,
        batch_size=1,
        dtype="float16",
    ):
        super().__init__()
        self.im_h, self.im_w = im_shape
        self.clip_box = clip_box
        self.nms_on = nms_on
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.iou_threshold = iou_threshold
        self.rpn_min_size = rpn_min_size
        self.level = level
        self.dtype = dtype
        self._feat_stride = feat_stride
        self.batch_size = batch_size
        self._scales = np.array(scales, dtype="float32")
        self._ratios = np.array(ratios, dtype="float32")
        if f_proc is None:
            f_proc = generate_shifted_anchors
        self._anchors = f_proc(
            self.im_h,
            self.im_w,
            self._feat_stride,
            self._scales,
            self._ratios,
            self.batch_size,
            self.dtype,
        )
        self._batch_inds = gen_batch_inds(
            batch_size,
            rpn_post_nms_top_n,
            dtype=dtype,
        )

    def forward(self, *args):
        assert len(args) >= 1
        N = self.batch_size

        x_bbox_deltas = args[0]
        bbox_deltas = ops.reshape()(x_bbox_deltas, [-1, 4])

        tensor_name = "anchors" if self.level == -1 else "anchors_%d" % (self.level)
        anchors = Tensor(shape=self._anchors.shape, dtype=self.dtype, name=tensor_name)

        proposals = self.box_transform(bbox_deltas, anchors)

        if self.nms_on:
            scores = args[1]
            scores_r = ops.reshape()(scores, [N, -1])
            proposals_r = ops.reshape()(proposals, [N, -1, 4])
            dets = ops.nms(
                self.rpn_pre_nms_top_n,
                self.rpn_post_nms_top_n,
                self.iou_threshold,
                self.rpn_min_size,
            )(proposals_r, scores_r)
            # prepare for roi-align for mask head
            batch_inds = Tensor(
                shape=[N, self.rpn_post_nms_top_n, 1],
                dtype=self.dtype,
                name="batch_inds",
                value=0,
            )
            ret = ops.reshape()(ops.concatenate()([batch_inds, dets], dim=2), [-1, 5])
            return ret, ops.reshape()(dets, [-1, 4])
        return proposals

    def box_transform(self, bbox_deltas, anchors):
        """apply transformation for proposals"""
        (delta_x, delta_y, delta_w, delta_h) = ops.split()(bbox_deltas, 1, dim=1)
        const_0_5 = Tensor(shape=[], dtype=self.dtype, name="const_0_5", value=0.5)

        (anchor_x1, anchor_y1, anchor_x2, anchor_y2) = ops.split()(anchors, 1, dim=1)
        widths = ops.elementwise(FuncEnum.SUB)(anchor_x2, anchor_x1)
        heights = ops.elementwise(FuncEnum.SUB)(anchor_y2, anchor_y1)

        width_mid = ops.elementwise(FuncEnum.MUL)(widths, const_0_5)
        height_mid = ops.elementwise(FuncEnum.MUL)(heights, const_0_5)
        ctr_x = ops.elementwise(FuncEnum.ADD)(anchor_x1, width_mid)
        ctr_y = ops.elementwise(FuncEnum.ADD)(anchor_y1, height_mid)

        pred_ctr_x = ops.elementwise(FuncEnum.ADD)(
            ops.elementwise(FuncEnum.MUL)(delta_x, widths),
            ctr_x,
        )
        pred_ctr_y = ops.elementwise(FuncEnum.ADD)(
            ops.elementwise(FuncEnum.MUL)(delta_y, heights),
            ctr_y,
        )
        pred_w = ops.elementwise(FuncEnum.MUL)(
            ops.elementwise(FuncEnum.EXP)(delta_w),
            widths,
        )
        pred_h = ops.elementwise(FuncEnum.MUL)(
            ops.elementwise(FuncEnum.EXP)(delta_h),
            heights,
        )

        p_x1 = ops.elementwise(FuncEnum.SUB)(
            pred_ctr_x,
            ops.elementwise(FuncEnum.MUL)(const_0_5, pred_w),
        )
        p_y1 = ops.elementwise(FuncEnum.SUB)(
            pred_ctr_y,
            ops.elementwise(FuncEnum.MUL)(const_0_5, pred_h),
        )
        p_x2 = ops.elementwise(FuncEnum.ADD)(
            pred_ctr_x,
            ops.elementwise(FuncEnum.MUL)(const_0_5, pred_w),
        )
        p_y2 = ops.elementwise(FuncEnum.ADD)(
            pred_ctr_y,
            ops.elementwise(FuncEnum.MUL)(const_0_5, pred_h),
        )

        if self.clip_box:
            x_min = Tensor(
                shape=[],
                dtype=self.dtype,
                name="X_min",
                value=0,
            )
            x_max_h = Tensor(
                shape=[],
                dtype=self.dtype,
                name="X_min_h",
                value=self.im_h,
            )
            x_max_w = Tensor(
                shape=[],
                dtype=self.dtype,
                name="X_min_w",
                value=self.im_w,
            )

            f_x1 = ops.elementwise(FuncEnum.HARDTANH)(p_x1, x_min, x_max_w)
            f_y1 = ops.elementwise(FuncEnum.HARDTANH)(p_y1, x_min, x_max_h)
            f_x2 = ops.elementwise(FuncEnum.HARDTANH)(p_x2, x_min, x_max_w)
            f_y2 = ops.elementwise(FuncEnum.HARDTANH)(p_y2, x_min, x_max_h)
            proposals = ops.concatenate()([f_x1, f_y1, f_x2, f_y2], dim=1)
        else:
            proposals = ops.concatenate()([p_x1, p_y1, p_x2, p_y2], dim=1)

        return proposals
