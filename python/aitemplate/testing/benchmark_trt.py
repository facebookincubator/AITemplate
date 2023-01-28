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
helper functions to benchmark fx-trt
"""
from aitemplate.testing.benchmark_pt import benchmark_torch_function  # usort:skip
from torch_tensorrt.fx import lower
from torch_tensorrt.fx.utils import LowerPrecision


def make_trt_module(
    function,
    *inputs,
    max_batch_size=256,
    max_workspace_size=2 << 31,
    dtype="float16",
):
    if dtype == "float16":
        lower_precision = LowerPrecision.FP16
    elif dtype == "float32":
        lower_precision = LowerPrecision.FP32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return lower.compile(
        function,
        inputs,
        min_acc_module_size=1,
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        lower_precision=lower_precision,
        verbose_log=True,
        timing_cache_prefix=True,
        save_timing_cache=True,
        explicit_batch_dimension=True,
        dynamic_batch=False,
    )


def benchmark_trt_function(iters: int, function, *args) -> float:
    submod = make_trt_module(function, args)
    submod(*args)
    return benchmark_torch_function(
        iters,
        submod,
        *args,
    )
