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

np.random.seed(0)
import unittest

import numpy.random as npr
import torch
from aitemplate.compiler import compile_model

from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import string_to_torch_dtype

DEBUG = False


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)

    return _nms


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack(
        [
            _scale_enum(ratio_anchors[i, :], scales)
            for i in range(ratio_anchors.shape[0])
        ]
    )
    return anchors


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    if DEBUG:
        print("clip boxes shape:", boxes.shape)
    # x1 >= 0
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], im_shape[1]), 0)
    # y1 >= 0
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], im_shape[0]), 0)
    # x2 < im_shape[1]
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], im_shape[1]), 0)
    # y2 < im_shape[0]
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], im_shape[0]), 0)
    return boxes


def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    # eltwise
    boxes = boxes.astype(np.float32, copy=False)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # slice
    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    # eltwise
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = np.hstack((x1, y1, x2, y2))

    return pred_boxes


class ProposalOP:
    def __init__(
        self,
        feat_stride,
        scales,
        ratios,
        output_score,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        threshold,
        rpn_min_size,
    ):
        self._feat_stride = feat_stride
        self._scales = np.array(scales, dtype="float32")
        self._ratios = np.array(ratios, dtype="float32")
        self._anchors = generate_anchors(
            base_size=self._feat_stride, scales=self._scales, ratios=self._ratios
        )
        self._num_anchors = self._anchors.shape[0]
        self._output_score = output_score
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size = rpn_min_size

        if DEBUG:
            print("feat_stride: {}".format(self._feat_stride))
            print("anchors:", self._anchors)

    def forward(self, in_data):
        nms = py_nms_wrapper(self._threshold)

        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size = self._rpn_min_size

        # the first set of anchors are background probabilities
        # keep the second part
        scores = in_data[0]
        bbox_deltas = in_data[1]
        im_info = in_data[2]

        if DEBUG:
            print("im_size: ({}, {})".format(im_info[0], im_info[1]))
            print("scale: {}".format(im_info[2]))

        # 1. Generate proposals from bbox_deltas and shifted anchors
        # use real image size instead of padded feature map sizes
        height, width = (
            int(im_info[0] / self._feat_stride),
            int(im_info[1] / self._feat_stride),
        )

        if DEBUG:
            print("score map size: {}".format(scores.shape))
            print(
                "residual: {}".format(
                    (scores.shape[2] - height, scores.shape[3] - width)
                )
            )

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack(
            (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
        ).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape(
            (1, K, 4)
        ).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # ----------- anchor can be pre-compuated ---------------

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order

        # may not need clip_pad if h/w are well aligned
        bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
        # permute + reshape
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = self._clip_pad(scores, (height, width))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = self._filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        det = np.hstack((proposals, scores)).astype(np.float32)
        keep = nms(det)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        if self._output_score:
            return blob, scores
        else:
            return blob

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """Remove all boxes with any side smaller than min_size"""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


class ProposalTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        test_name="proposal",
        dtype="float16",
    ):
        target = detect_target()
        feat_stride = 16
        scales = [128, 256, 512]
        ratios = [0.5, 1, 2]
        num_anchors = len(scales) * len(ratios)

        output_score = False
        rpn_pre_nms_top_n = 6000
        rpn_post_nms_top_n = 300
        threshold = 0.7
        rpn_min_size = 0
        op = ProposalOP(
            feat_stride,
            scales,
            ratios,
            output_score,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            threshold,
            rpn_min_size,
        )
        batch_size = 2
        in_data = [
            npr.rand(1, num_anchors, 32, 32),
            npr.rand(1, num_anchors * 4, 32, 32),
            np.array([512, 512, 1.0]),
        ]
        _ = op.forward(in_data)

        scores = in_data[0]
        bbox_deltas = in_data[1]
        im_info = in_data[2]

        bbox_deltas = np.repeat(bbox_deltas, repeats=batch_size, axis=0)
        scores = np.repeat(scores, repeats=batch_size, axis=0)

        bbox_deltas_ait = np.transpose(
            bbox_deltas.astype(dtype),
            (0, 2, 3, 1),
        ).copy()
        scores_ait = np.transpose(
            scores.astype(dtype),
            (0, 2, 3, 1),
        ).copy()

        X_bbox_deltas = Tensor(
            shape=bbox_deltas_ait.shape,
            name="X_bbox_deltas",
            dtype=dtype,
            is_input=True,
        )
        X_scores = Tensor(
            shape=scores_ait.shape,
            name="X_scores",
            dtype=dtype,
            is_input=True,
        )

        OP = nn.Proposal(
            im_shape=im_info[:2],
            scales=scales,
            ratios=ratios,
            clip_box=True,
            nms_on=True,
            rpn_pre_nms_top_n=rpn_pre_nms_top_n,
            rpn_post_nms_top_n=rpn_post_nms_top_n,
            iou_threshold=threshold,
            rpn_min_size=rpn_min_size,
            batch_size=batch_size,
            dtype=dtype,
        )

        y = OP(X_bbox_deltas, X_scores)
        mark_output(y)
        module = compile_model(y, target, "./tmp", test_name)

        anchors = torch.from_numpy(OP._anchors.copy()).cuda()
        batch_inds = torch.from_numpy(OP._batch_inds.copy()).cuda()
        module.set_constant_with_tensor("anchors", anchors)
        module.set_constant_with_tensor("batch_inds", batch_inds)
        torch_dtype = string_to_torch_dtype(dtype)
        inputs_pt = [
            torch.from_numpy(bbox_deltas_ait).cuda().to(torch_dtype),
            torch.from_numpy(scores_ait).cuda().to(torch_dtype),
        ]
        out0_shape = module.get_output_maximum_shape(0)
        out0 = torch.empty(out0_shape, dtype=torch_dtype, device="cuda")
        y_ait_shape = module.get_output_maximum_shape(1)
        y_ait = torch.empty(y_ait_shape, dtype=torch_dtype, device="cuda")
        module.run_with_tensors(inputs_pt, [out0, y_ait])
        y_ait = y_ait.reshape(2, -1, 4)
        self.assertTrue(torch.allclose(y_ait[0, :], y_ait[1, :], atol=1e-2, rtol=1e-2))

    def test_proposal_fp16(self):
        self._test_single_op(
            test_name="proposal_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_proposal_fp32(self):
        self._test_single_op(
            test_name="proposal_fp32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
