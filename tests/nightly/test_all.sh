#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

bash /AITemplate/tests/nightly/benchmark.sh
bash /AITemplate/tests/nightly/unittest.sh
bash /AITemplate/benchmark/detection/compile_bench_run.sh
