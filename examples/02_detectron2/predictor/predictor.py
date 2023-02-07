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
import itertools
import os
from typing import Tuple

import cv2
import numpy as np
import torch

from aitemplate.compiler import Model
from modeling.meta_arch import GeneralizedRCNN
from PIL import Image

from .builtin_meta import _get_coco_instances_meta


class Predictor:
    """
    Use this class to create AIT inference instances for detectron2 models. It includes procedures that is to 1) preprocess the input images, 2) load the weights of the AIT model, 3) run the AIT model and visualize the outputs, 4) benchmark the AIT model.
    """

    def __init__(self, cfg, workdir="./tmp"):
        self.cfg = cfg
        self.model_name = cfg.MODEL.NAME
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.im_shape = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        self.pixel_mean = cfg.MODEL.PIXEL_MEAN
        self.pixel_std = cfg.MODEL.PIXEL_STD
        self.mask_on = cfg.MODEL.MASK_ON
        self.model = GeneralizedRCNN(cfg)
        self.weights = self.get_parameters()
        self.module = self.init_modules(cfg.MODEL.NAME, workdir)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.min_size = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST
        self.interp_method = Image.BILINEAR

    def get_parameters(self):
        """
        Obtain the weights.
        """
        parameters = {
            name: w.contiguous().cuda().half()
            for name, w in torch.load(self.cfg.MODEL.WEIGHTS).items()
        }
        for name, param in self.model._params.items():
            parameters[name] = torch.from_numpy(param).cuda().half()
        return parameters

    def preprocess(self, im_path: str, pad_value: float = 0.0):
        """
        Image preprocess: resize the image (see `apply_transform`), normalize the pixels,
        and add padding.
        """
        # HH, WW = self.im_shape
        ori_img = cv2.imread(im_path)
        ori_shape = ori_img.shape
        if ori_shape[0] > ori_shape[1]:
            img = np.rot90(ori_img, k=1)
        else:
            img = ori_img
        inputs = self.apply_transform(img)
        resize_scale = img.shape[0] / inputs.shape[0]
        pixel_mean = np.array(self.pixel_mean).reshape(1, 1, -1)
        pixel_std = np.array(self.pixel_std).reshape(1, 1, -1)
        inputs = (inputs - pixel_mean) / pixel_std
        padding_size = (
            (0, self.min_size - inputs.shape[0]),
            (0, self.max_size - inputs.shape[1]),
            (0, 0),
        )
        inputs = np.pad(inputs, padding_size, constant_values=pad_value)
        inputs = inputs[np.newaxis, :]
        return inputs.astype("float16"), ori_img, ori_shape, resize_scale

    def apply_transform(self, img):
        """
        Resize the image while keeping the aspect ratio unchanged.
        It attempts to scale the shorter edge to the given `short_edge_length`,
        as long as the longer edge does not exceed `max_size`.
        If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
        """
        h, w = img.shape[:2]
        new_h, new_w = Predictor.get_output_shape(h, w, self.min_size, self.max_size)
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((new_w, new_h), self.interp_method)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_bbox(self, bbox, im_w, im_h):
        if im_h > im_w:
            x0 = bbox[:, 0][..., np.newaxis]
            y0 = bbox[:, 1][..., np.newaxis]
            x1 = bbox[:, 2][..., np.newaxis]
            y1 = bbox[:, 3][..., np.newaxis]
            bbox = np.hstack((im_w - y1, x0, im_w - y0, x1))
        return bbox

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def data_loader(self, image_list):
        """
        Load the images and convert them to batched data.
        """
        batch_data = []
        HH, WW = self.im_shape
        batch = np.zeros((self.batch_size, HH, WW, 3), dtype="float16")
        img_paths, img_shapes, img_scales, raw_images = [], [], [], []
        num_samples = len(image_list)
        max_iter = (
            (num_samples + self.batch_size - 1) // self.batch_size * self.batch_size
        )
        datasets = itertools.cycle(image_list)
        for idx in range(max_iter):
            im_path = next(datasets)
            input_data, raw_input, im_shape, im_scale = self.preprocess(im_path)
            im_name = im_path.split("/")[-1]
            img_paths.append(im_name)
            img_shapes.append(im_shape)
            img_scales.append(im_scale)
            raw_images.append(raw_input)
            batch[idx % self.batch_size, :, :, :] = input_data
            if (idx + 1) % self.batch_size == 0:
                batch_data.append(
                    {
                        "data": batch.astype("float16"),
                        "image_shape": img_shapes,
                        "image_scale": img_scales,
                        "path": img_paths,
                        "image": raw_images,
                    }
                )
                img_paths, img_shapes, img_scales, raw_images = [], [], [], []
        return batch_data

    def init_modules(self, detection_model_name, workdir):
        """
        Load the AIT module of the detection model, and set the weights.
        """
        mod = Model(os.path.join(workdir, detection_model_name, "test.so"))
        mod.set_many_constants_with_tensors(self.weights)
        mod.fold_constants(sync=True)
        return mod

    def run_batch(self, batch_data, graph_mode=False):
        """
        Run the inference of the AIT model with batched data.
        """
        score_thresh = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        results = {}
        inputs = batch_data["data"]
        image_list = batch_data["path"]
        image_shapes = batch_data["image_shape"]
        image_scales = batch_data["image_scale"]
        images = batch_data["image"]
        ret = self.run_on_image(inputs, graph_mode=graph_mode)
        batched_boxes, batched_scores, batched_classes = ret[:3]
        if self.mask_on:
            batched_masks = ret[-1]
        for i in range(self.batch_size):
            boxes, scores, classes = (
                batched_boxes[i, :],
                batched_scores[i, :],
                batched_classes[i, :],
            )

            filter_mask = scores > score_thresh
            filter_inds = filter_mask.nonzero()[0]
            scores = scores[filter_inds]
            boxes = boxes[filter_inds, :] * image_scales[i]
            boxes = self.apply_bbox(boxes, image_shapes[i][1], image_shapes[i][0])
            classes = classes[filter_inds]

            results[image_list[i]] = {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "image_height": image_shapes[i][0],
                "image_width": image_shapes[i][1],
                "num_instances": boxes.shape[0],
                "image": images[i],
            }
            if self.mask_on:
                mask_pred = batched_masks[i, filter_inds, :, :]
                results[image_list[i]]["masks"] = mask_pred
        return results

    @staticmethod
    def overlay(image, mask, color, alpha_transparency=0.5):
        for channel in range(3):
            image[:, :, channel] = np.where(
                mask == 1,
                image[:, :, channel] * (1 - alpha_transparency)
                + alpha_transparency * color[channel] * 255,
                image[:, :, channel],
            )
        return image

    def visualize(
        self, detections, output_path="./tmp/outputs", thickness=1, mask_thresh=0.5
    ):
        """
        Visualize the outputs.
        """
        os.makedirs(output_path, exist_ok=True)
        meta_data = _get_coco_instances_meta()
        thing_colors = meta_data["thing_colors"]
        thing_classes = meta_data["thing_classes"]
        for file_name, result in detections.items():
            img = result["image"]
            boxes = result["boxes"]
            classes = result["classes"]
            scores = result["scores"]
            for pred_box, pred_class, pred_score in zip(boxes, classes, scores):
                box = pred_box.astype("int")
                start_point = (box[0], box[1])
                end_point = (box[2], box[3])
                color = tuple(thing_colors[pred_class])
                img = cv2.rectangle(img, start_point, end_point, color, thickness)
                text = thing_classes[pred_class] + ": " + str(pred_score)
                img = cv2.putText(
                    img,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    thickness,
                )

            if self.mask_on:
                masks = result["masks"]
                im_height, im_width = img.shape[:2]
                for pred_box, pred_class, mask in zip(boxes, classes, masks):
                    np_color = np.array(thing_colors[pred_class]) / 255
                    if im_height > im_width:
                        mask = np.rot90(mask, k=-1)
                    box = pred_box.astype("int")
                    det_width = box[2] - box[0]
                    det_height = box[3] - box[1]
                    mask = mask.astype(np.float32)
                    small_mask = Image.fromarray(mask)
                    mask = small_mask.resize(
                        (det_width, det_height), resample=self.interp_method
                    )
                    mask = np.array(mask, copy=False)
                    mask = np.array(mask > mask_thresh, dtype=np.uint8)
                    padded_mask = np.zeros((im_height, im_width), dtype=np.uint8)
                    x_0 = max(box[0], 0)
                    x_1 = min(box[2], im_width)
                    y_0 = max(box[1], 0)
                    y_1 = min(box[3], im_height)
                    padded_mask[y_0:y_1, x_0:x_1] = mask[
                        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
                    ]
                    img = Predictor.overlay(img, padded_mask, np_color)
            cv2.imwrite(os.path.join(output_path, file_name), img)

    def run_on_image(self, inputs, graph_mode=False):
        """
        Call the AIT module for the inference of the model on given inputs, and return the outputs.
        """
        topk = self.cfg.POSTPROCESS.TOPK
        mod = self.module
        if type(inputs) is np.ndarray:
            arr = torch.from_numpy(inputs).cuda()
        else:
            arr = inputs.contiguous()

        inputs = [arr]

        outputs = [
            torch.empty([self.batch_size, 1], dtype=torch.int64).cuda(),
            torch.empty([self.batch_size, topk, 4]).cuda().half(),
            torch.empty([self.batch_size, topk]).cuda().half(),
            torch.empty([self.batch_size, topk], dtype=torch.int64).cuda(),
        ]
        if self.mask_on:
            mask_size = self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
            mask_blob = torch.empty([self.batch_size, topk, mask_size, mask_size])
            outputs.append(mask_blob.cuda().half())
        mod.run_with_tensors(inputs, outputs, graph_mode=graph_mode)

        ret = [
            outputs[1].cpu().numpy(),
            outputs[2].cpu().numpy(),
            outputs[3].cpu().numpy(),
        ]
        if self.mask_on:
            ret.append(outputs[-1].cpu().numpy())
        return ret

    def benchmark(self, inputs, count=10, graph_mode=False):
        """
        Benchmark the inference of the AIT model on given inputs, and return the runtime in ms.
        """
        mod = self.module
        if type(inputs) is np.ndarray:
            arr = torch.from_numpy(inputs).cuda()
        else:
            arr = inputs.cuda().contiguous()
        topk = self.cfg.POSTPROCESS.TOPK
        outputs = [
            torch.empty([self.batch_size, 1], dtype=torch.int64).cuda(),
            torch.empty([self.batch_size, topk, 4]).cuda().half(),
            torch.empty([self.batch_size, topk]).cuda().half(),
            torch.empty([self.batch_size, topk], dtype=torch.int64).cuda(),
        ]
        if self.mask_on:
            mask_blob = torch.empty([self.batch_size, topk, 28, 28])
            outputs.append(mask_blob.cuda().half())

        duration, _, _ = mod.benchmark_with_tensors(
            [arr],
            outputs,
            count=count,
            repeat=2,
            graph_mode=graph_mode,
        )
        return duration
