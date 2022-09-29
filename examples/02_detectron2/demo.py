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
A main inference script for rcnn models
"""
import glob
import os

import click
import tqdm
from configs import get_cfg_defaults
from predictor import Predictor


@click.command()
@click.option("--config", default="", metavar="FILE", help="path to config file")
@click.option("--bench-config", default="", metavar="FILE", help="path to config file")
@click.option(
    "--input",
    multiple=True,
    help="A list of space separated input images; "
    "or a single glob pattern such as 'directory/*.jpg'",
)
@click.option(
    "--output",
    help="A file or directory to save output visualizations. "
    "If not given, will show output in an OpenCV window.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown",
)
@click.option("--weight", default="", metavar="FILE", help="path to model weights")
@click.option("--batch", default=0, help="batch size")
@click.option("--display/--no-display", default=False, help="display results")
@click.option("--cudagraph/--no-cudagraph", default=False, help="enable CUDA graph")
def run_model(
    config,
    bench_config,
    input,
    output,
    confidence_threshold,
    weight,
    batch,
    display,
    cudagraph,
):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    if bench_config != "":
        cfg.merge_from_file(bench_config)
    if batch > 0:
        cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.MODEL.WEIGHTS = weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.freeze()

    assert (
        weight != ""
    ), "export model first: python convert_pt2ait.py model_d2.pkl params_ait.pkl \
    --config configs/faster_rcnn_R_50_DC5.yaml"

    demo = Predictor(cfg)
    print("run {} end2end".format(cfg.MODEL.NAME))

    cnt = 0
    duration = 0
    detections = {}
    bs = cfg.SOLVER.IMS_PER_BATCH
    if input:
        if len(input) == 1:
            input = glob.glob(os.path.expanduser(input[0]))
            assert input, "The input path(s) was not found"
        batch_data = demo.data_loader(input)
        print("{} images, run {} batch".format(len(input), len(batch_data)))
        for batch in tqdm.tqdm(batch_data, disable=not output):
            results = demo.run_batch(batch, cudagraph)
            detections.update(results)
            if display:
                demo.visualize(results)
            duration += demo.benchmark(batch["data"], 10, cudagraph)
            cnt += 1

    duration /= cnt * bs
    print(
        f"AIT Detection: Batch size: {bs}, Time per iter: {duration:.2f} ms, FPS: {1000 / duration:.2f}"
    )


if __name__ == "__main__":
    run_model()
