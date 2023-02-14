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
import functools
import logging
import os
import unittest

import cv2
import numpy as np

import torch

from aitemplate.compiler import compile_model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from detectron2.config import CfgNode
from detectron2.engine import DefaultPredictor

try:
    from libfb.py.asyncio.await_utils import await_sync
    from manifold.clients.python import ManifoldClient
except ImportError:
    ManifoldClient = None
    import requests

from detectron2.model_zoo import get_checkpoint_url
from parameterized import parameterized
from PIL import Image

from .configs.config import get_cfg_defaults
from .modeling.meta_arch import GeneralizedRCNN
from .tools.convert_pt2ait import detectron2_export

logger = logging.getLogger(__name__)


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def extract_params_meta(ait_model):
    ret = []
    for name, p in ait_model.named_parameters():
        name = name.replace(".", "_")
        shape = [x._attrs["values"][0] for x in p.tensor()._attrs["shape"]]
        ret.append([name, shape])
    return ret


def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
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


def apply_transform(cfg, img):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """
    h, w = img.shape[:2]
    new_h, new_w = get_output_shape(
        h, w, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
    )
    if len(img.shape) > 2 and img.shape[2] == 1:
        pil_image = Image.fromarray(img[:, :, 0], mode="L")
    else:
        pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
    ret = np.asarray(pil_image)
    if len(img.shape) > 2 and img.shape[2] == 1:
        ret = np.expand_dims(ret, -1)
    return ret


def preprocess(cfg, ori_img, pad_value: float = 0.0, dtype="float16"):
    """
    Image preprocess: resize the image (see `apply_transform`), normalize the pixels,
    and add padding.
    """
    # HH, WW = self.im_shape
    ori_shape = ori_img.shape
    if ori_shape[0] > ori_shape[1]:
        img = np.rot90(ori_img, k=1)
    else:
        img = ori_img
    inputs = apply_transform(cfg, img)
    resize_scale = img.shape[0] / inputs.shape[0]
    pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(1, 1, -1)
    pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(1, 1, -1)
    inputs = (inputs - pixel_mean) / pixel_std
    padding_size = (
        (0, cfg.INPUT.MIN_SIZE_TEST - inputs.shape[0]),
        (0, cfg.INPUT.MAX_SIZE_TEST - inputs.shape[1]),
        (0, 0),
    )
    inputs = np.pad(inputs, padding_size, constant_values=pad_value)
    inputs = inputs[np.newaxis, :]
    return inputs.astype(dtype), ori_img, ori_shape, resize_scale


def apply_bbox(bbox, im_w, im_h):
    if im_h > im_w:
        x0 = bbox[:, 0][..., np.newaxis]
        y0 = bbox[:, 1][..., np.newaxis]
        x1 = bbox[:, 2][..., np.newaxis]
        y1 = bbox[:, 3][..., np.newaxis]
        bbox = np.hstack((im_w - y1, x0, im_w - y0, x1))
    return bbox


def postprocess_ait_results(
    ret,
    mask_on,
    batch_size,
    score_thresh,
    images,
    image_list,
    image_shapes,
    image_scales,
):
    batched_boxes, batched_scores, batched_classes = ret[1:4]
    if mask_on:
        batched_masks = ret[-1]
    results = {}
    for i in range(batch_size):
        boxes, scores, classes = (
            batched_boxes[i, :],
            batched_scores[i, :],
            batched_classes[i, :],
        )

        filter_inds = (scores > score_thresh).nonzero().squeeze()
        scores = scores[filter_inds]
        boxes = boxes[filter_inds, :] * image_scales[i]
        boxes = apply_bbox(boxes, image_shapes[i][1], image_shapes[i][0])
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
        if mask_on:
            mask_pred = batched_masks[i, filter_inds, :, :]
            im_height, im_width = image_shapes[i][:2]
            masks = []
            for pred_box, mask in zip(
                boxes,
                mask_pred,
            ):
                mask = mask.cpu().numpy().astype(np.float32)
                if im_height > im_width:
                    mask = np.rot90(mask, k=-1)
                box = pred_box.cpu().numpy().astype("int")
                det_width = box[2] - box[0]
                det_height = box[3] - box[1]
                small_mask = Image.fromarray(mask)
                mask = small_mask.resize(
                    (det_width, det_height), resample=Image.BILINEAR
                )
                mask = np.array(mask, copy=False)
                MASK_THRESHOLD = 0.5
                mask = np.array(mask > MASK_THRESHOLD, dtype=np.uint8)
                padded_mask = np.zeros((im_height, im_width), dtype=np.uint8)
                x_0 = max(box[0], 0)
                x_1 = min(box[2], im_width)
                y_0 = max(box[1], 0)
                y_1 = min(box[3], im_height)
                padded_mask[y_0:y_1, x_0:x_1] = mask[
                    (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
                ]
                masks.append(padded_mask)
            results[image_list[i]]["masks"] = torch.tensor(masks)
    return results


class Detectron2Verification(unittest.TestCase):
    @parameterized.expand(
        ["faster_rcnn_R_50", "faster_rcnn_R_101", "mask_rcnn_R_50", "mask_rcnn_R_101"]
    )
    def test_detectron2(self, config):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(
            os.path.join(os.path.dirname(__file__), "configs", f"{config}_FPN.yaml")
        )
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.freeze()

        ait_dtype = "float16"
        torch_dtype = torch.float16

        model = GeneralizedRCNN(cfg)
        model.name_parameter_tensor()

        x = Tensor(
            shape=[
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                3,
            ],
            dtype=ait_dtype,
            name="input_0",
            is_input=True,
        )
        y = model(x)
        mark_output(y)

        checkpoint_path = f"/tmp/detectron2/{config}_FPN_3x.pkl"
        sample_input_filename = "000000001268.jpg"
        sample_input_path = f"/tmp/detectron2/{sample_input_filename}"

        torch_cfg = CfgNode(cfg)
        torch_cfg.MODEL.WEIGHTS = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            if ManifoldClient is not None:
                with ManifoldClient.get_client("glow_test_data") as client:
                    await_sync(
                        client.get(
                            f"tree/aitemplate/detectron2/pickles/{config}_FPN_3x.pkl",
                            checkpoint_path,
                        )
                    )
            else:
                torch_cfg.MODEL.WEIGHTS = get_checkpoint_url(
                    f"COCO-{'InstanceSegmentation' if 'mask' in config else 'Detection'}/{config}_FPN_3x.yaml"
                )

        torch_predictor = DefaultPredictor(torch_cfg)

        if not os.path.exists(sample_input_path):
            if ManifoldClient is not None:
                with ManifoldClient.get_client("aitemplate") as client:
                    await_sync(
                        client.get(
                            f"tree/detectron2/datasets/coco/val2017/{sample_input_filename}",
                            sample_input_path,
                        )
                    )
            else:
                img_url = (
                    f"http://images.cocodataset.org/val2017/{sample_input_filename}"
                )
                img_data = requests.get(img_url).content
                with open(sample_input_path, "wb") as f:
                    f.write(img_data)

        sample_img = cv2.imread(sample_input_path)
        sample_input, original_image, shape, scale = preprocess(
            cfg, sample_img, dtype=ait_dtype
        )
        x_ait = torch.tensor(sample_input).cuda()

        with torch.no_grad():
            ait_params = detectron2_export("").export_model(
                {
                    k: v.cpu().numpy()
                    for k, v in torch_predictor.model.state_dict().items()
                },
                extract_params_meta(model),
            )
            pt_instance = torch_predictor(sample_img)["instances"]

        ait_module = compile_model(y, detect_target(), "./tmp", cfg.MODEL.NAME)
        for name, param in ait_params.items():
            ait_module.set_constant_with_tensor(
                name, param.contiguous().to(dtype=torch_dtype).cuda()
            )
        model.set_anchors(ait_module)
        topk = cfg.POSTPROCESS.TOPK
        BS = cfg.SOLVER.IMS_PER_BATCH
        outputs = [
            torch.empty([BS, 1], dtype=torch.int64).cuda(),
            torch.empty([BS, topk, 4], dtype=torch_dtype).cuda(),
            torch.empty([BS, topk], dtype=torch_dtype).cuda(),
            torch.empty([BS, topk], dtype=torch.int64).cuda(),
        ]
        if cfg.MODEL.MASK_ON:
            mask_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
            outputs.append(
                torch.empty([BS, topk, mask_size, mask_size], dtype=torch_dtype).cuda()
            )

        ait_module.run_with_tensors([x_ait], outputs)

        ait_results = postprocess_ait_results(
            outputs,
            cfg.MODEL.MASK_ON,
            BS,
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            [original_image],
            [sample_input_path],
            [shape],
            [scale],
        )

        result = ait_results[sample_input_path]

        compare_floats = functools.partial(
            torch.testing.assert_close, atol=1e-1, rtol=1e-1
        )
        compare_ints = functools.partial(torch.testing.assert_close, atol=0, rtol=0)

        compare_ints(len(pt_instance), result["num_instances"])

        # Boxes precision is tricky.
        # Practically, these are pixel values, so any difference around 1e0 can be disregarded
        compare_boxes_floats = functools.partial(
            torch.testing.assert_close, atol=5e-0, rtol=1e-1
        )
        # Keep in mind that we are comparing sets here,
        # not lists because all items are sorted by score and
        # a small difference in score can result in a wrong items order.
        # We do our best to estabilish 1:1 mapping for comparison
        pt_boxes = pt_instance.pred_boxes.tensor.to(dtype=result["boxes"].dtype).sort(
            dim=0
        )
        ait_boxes = result["boxes"].sort(dim=0)
        compare_boxes_floats(
            ait_boxes,
            pt_boxes,
        )
        compare_floats(
            pt_instance.scores.to(dtype=result["scores"].dtype),
            result["scores"],
        )
        # also comparing sets
        compare_ints(
            pt_instance.pred_classes.sort().values, result["classes"].sort().values
        )
        # homebrew similarity match between boolean arrays
        if cfg.MODEL.MASK_ON:
            pt_masks = pt_instance.pred_masks.to(
                dtype=result["masks"].dtype, device="cpu"
            )
            ait_masks = result["masks"]
            self.assertLess(
                (pt_masks != ait_masks).sum() / (pt_masks == ait_masks).sum(), 1e-2
            )


if __name__ == "__main__":
    torch.cuda.manual_seed(1337)
    unittest.main()
