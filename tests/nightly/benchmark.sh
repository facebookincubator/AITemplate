#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/inline_cvr_7x/benchmark_model.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/inline_cvr_7x/verification_cuda.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/ctr_mbl_feed_30x/benchmark_model.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/ctr_mbl_feed_30x/verification_cuda.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/umia_v1/benchmark_model.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/umia_v1/verification_cuda.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/resnet_50/model_relu_cuda_def.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/resnet_50/model_relu_cuda_def.py -graph_mode
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NIGHTLY=1 PYTHONPATH=/AITemplate python3 /AITemplate/benchmark/resnet_50/model_relu_cuda_def.py -verify

echo "JSON_REPORT_START"
echo "================="
echo ""

cat /AITemplate/benchmark_result.json

echo ""
echo "================="
echo "JSON_REPORT_END"
