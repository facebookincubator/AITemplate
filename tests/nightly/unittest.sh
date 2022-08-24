#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# NUM_BUILDERS=12 python3 /AITemplate/tests/nightly/test_runner.py
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest --junitxml="/AITemplate/unittest_result.xml" /AITemplate/tests/unittest

echo "XML_REPORT_START"
echo "================"
echo ""

cat /AITemplate/unittest_result.xml

echo ""
echo "==============="
echo "XML_REPORT_END"
