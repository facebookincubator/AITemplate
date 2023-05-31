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
Torch module profiling utility.
"""
import logging
from operator import itemgetter
from typing import Callable, List, Tuple

import torch

logger = logging.getLogger(__name__)


def profile_callable(
    func: Callable,
    cache_flush_slab: torch.Tensor,
    n_iter: int,
) -> Tuple[List[int], List[int]]:
    """
    Profile the callable and return the device and wall time for each iteration.
    We assume the iterations happen sequentially, not concurrently.
    Example usage:
    .. code-block:: python
        x = torch.randn((4096, 2048), device='cuda')
        y = torch.randn((8192, 2048), device='cuda')
        xy = torch.empty((4096, 8192), device='cuda')
        slab = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device='cuda')
        def _f():
            torch.nn.functional.linear(x, y, out=xy)
        profile_callable(_f, slab, 100)
    Parameters
    ----------
    func: Callable
        The callable to profile.
    cache_flush_slab: torch.Tensor
        A slab of GPU memory. We flush the device L2 cache by filling the slab.
    n_iter: int
        The number of iterations to call the callable.
    Returns
    -------
        device_times: List[int]
            Sum of the kernel device times (µs) for each iteration.
        wall_times: List[int]
            Times (µs) from the start of the first kernel
            until the end of the last kernel for each iteration.
    """
    if n_iter <= 0:
        return [], []
    # warmup
    for _ in range(5):
        func()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(n_iter):
            cache_flush_slab.fill_(3.7)
            func()
    # log the invoked kernels
    results = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        max_name_column_width=120,
        row_limit=-1,
    )
    logger.info(results)

    events = [
        {
            "name": e.name,
            "cuda_time": e.cuda_time,
            "start": e.time_range.start,
            "end": e.time_range.end,
        }
        for e in prof.events()
        if e.cuda_time != 0
    ]

    sorted_events = sorted(events, key=itemgetter("start"))
    assert 0 == len(sorted_events) % n_iter
    n_groups = len(sorted_events) // n_iter
    # in each group (corresponding to a profiling iteration),
    # skip measuring the first kernel, which is the l2 cache flush
    event_groups = [g[1:] for g in zip(*([iter(sorted_events)] * n_groups))]
    logger.info(
        f"First kernel sequence: {list(map(itemgetter('name'), event_groups[0]))}"
    )
    device_times = [sum(map(itemgetter("cuda_time"), g)) for g in event_groups]
    wall_times = [
        g[-1]["end"] - g[0]["start"] if len(g) > 0 else 0 for g in event_groups
    ]
    return device_times, wall_times
