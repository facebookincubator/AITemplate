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
helper function to benchmark eager pytorch
"""
# pylint: disable=C0415


def benchmark_torch_function(iters: int, function, *args, **kwargs) -> float:
    """
    function for benchmarking a pytorch function.

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
        function(*args, **kwargs)

    # Start benchmark.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        function(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # in ms
    return (start_event.elapsed_time(end_event)) / iters
