# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] utils for unit tests
"""
from typing import Dict

import torch

DTYPE_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "float16": torch.half,
    "float": torch.float,
    "int": torch.int,
}


def dtype_to_torch_dtype(dtype):
    if dtype is None:
        return None
    torch_dtype = DTYPE_TO_TORCH_DTYPE.get(dtype)
    if torch_dtype is None:
        raise RuntimeError("Unsupported dtype: {}".format(dtype))
    return torch_dtype


def get_random_torch_tensor(shape, dtype):
    if dtype == "float16":
        return torch.randn(shape).cuda().half()
    if dtype == "float":
        return torch.randn(shape).cuda().float()
    if dtype == "int":
        return torch.randn(shape).cuda().int()
    raise RuntimeError("unsupported dtype: {}".format(dtype))


def benchmark_torch_function(iters: int, function, *args) -> float:
    """
    function for benchmarking a pytorch function
    """
    # Warm up
    for _ in range(5):
        function(*args)

    # Start benchmark.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        function(*args)
    end_event.record()
    torch.cuda.synchronize()
    # in ms
    return (start_event.elapsed_time(end_event)) / iters
