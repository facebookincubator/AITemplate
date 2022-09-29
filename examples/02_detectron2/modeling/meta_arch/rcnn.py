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
import torch
from aitemplate.frontend import nn
from aitemplate.frontend.nn.proposal import gen_batch_inds

from ..backbone import build_resnet_fpn_backbone
from ..proposal_generator import build_rpn_head
from ..roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        im_shape = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        self._batch_size = cfg.SOLVER.IMS_PER_BATCH
        self._mask_on = cfg.MODEL.MASK_ON
        self._num_mask_roi = cfg.POSTPROCESS.TOPK

        self.backbone = build_resnet_fpn_backbone(cfg)
        self.proposal_generator = build_rpn_head(cfg, im_shape)
        self.roi_heads = build_roi_heads(cfg, im_shape)
        self._params = self.get_params()

    def forward(self, x):
        features = self.backbone(x)
        rois, proposals = self.proposal_generator(features)
        results = self.roi_heads(features, rois, proposals)
        return results

    def set_anchors(self, mod):
        self.proposal_generator.set_anchors(mod)
        if self._mask_on:
            batch_inds_mask = gen_batch_inds(self._batch_size, self._num_mask_roi)
            weight = torch.from_numpy(batch_inds_mask).cuda().half()
            mod.set_constant_with_tensor("batch_inds_mask", weight)

    def get_params(self):
        params = self.proposal_generator.get_params()
        if self._mask_on:
            params["batch_inds_mask"] = gen_batch_inds(
                self._batch_size, self._num_mask_roi
            )
        return params
