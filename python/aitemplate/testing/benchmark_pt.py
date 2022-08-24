# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] helper function to benchmark eager pytorch
"""
# pylint: disable=C0415


def benchmark_torch_function(iters: int, function, *args) -> float:

    """[summary]
    Benchmark pytorch model.

    Parameters
    ----------
    iters: int
        Number of iterations.
    function: lambda function
        function to benchmark.
    args: Any type
        Args to function.

    Returns
    -------
    float
        Runtime per iteration in ms.
    """
    import torch

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
    return start_event.elapsed_time(end_event) / iters
