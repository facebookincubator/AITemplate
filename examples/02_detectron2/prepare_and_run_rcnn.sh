#!/bin/bash -e
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

### Download COCO 2017 Dataset 

#### Download image annotations
BASE=https://dl.fbaipublicfiles.com/detectron2
ROOT=~/.torch/datasets
mkdir -p $ROOT/coco/annotations
echo "$ROOT"

for anno in instances_val2017_100 \
  person_keypoints_val2017_100 ; do

  dest=$ROOT/coco/annotations/$anno.json
  [[ -s $dest ]] && {
    echo "$dest exists. Skipping ..."
  } || {
    wget $BASE/annotations/coco/$anno.json -O $dest
  }
done

#### Download images
dest=$ROOT/coco/val2017_100.tgz
[[ -d $ROOT/coco/val2017 ]] && {
  echo "$ROOT/coco/val2017 exists. Skipping ..."
} || {
  wget $BASE/annotations/coco/val2017_100.tgz -O $dest
  tar xzf $dest -C $ROOT/coco/ && rm -f $dest
}
IMG_PATH=$ROOT/coco/val2017

### Download Pre-trained Model

MODEL_PATH=~/.torch/model
mkdir -p $MODEL_PATH
MODEL_NAME=mask_rcnn_R_50_FPN

mkdir -p ./tmp

wget $BASE/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -O tmp/pt_$MODEL_NAME.pkl

### Build AIT Model, Export the Pre-trained Weights and Run Inference 

cfg=configs/$MODEL_NAME.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 compile_model.py   \
  --config $cfg \
  --batch 1

python3 tools/convert_pt2ait.py  \
  --d2-weight tmp/pt_$MODEL_NAME.pkl \
  --ait-weight tmp/ait_$MODEL_NAME.pt \
  --model-name $MODEL_NAME

python3 demo.py \
  --weight tmp/ait_$MODEL_NAME.pt \
  --config $cfg \
  --batch 1 --input "$IMG_PATH/*.jpg" \
  --confidence-threshold 0.5 \
  --display \
  --cudagraph
