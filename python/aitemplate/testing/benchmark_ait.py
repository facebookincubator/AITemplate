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

from typing import Optional

import torch


def make_input_output_pools(
    *, pool_size, eval_pt_func, input_filter_func, output_filter_func
):
    """
    Make input and output pools for benchmarking. The rationale is avoiding retrieving the same input from the device cache for fair perf assessment.
    Parameters
    ----------
    pool_size : int
        The size of the pool.
    eval_pt_func : callable
        A callable that returns a dict of inputs and outputs.
    input_filter_func : callable
        A callable that takes a key and a value and returns True if the key-value pair from the `eval_pt_func` result should be included in the input pool.
    output_filter_func : callable
        A callable that takes a key and a value and returns True if the key-value pair from the `eval_pt_func` result should be included in the output pool.
    Returns
    -------
    inputs_pool : List[Dict[str, torch.Tensor]]
        A list of inputs to pass into Model.RunWithTensors.
    outputs_pool : List[Dict[str, torch.Tensor]]
        A list of outputs to pass into Model.RunWithTensors.
    """
    return zip(
        *[
            [
                {k: v for k, v in d.items() if input_filter_func(k, v)},
                {
                    k: torch.empty_like(v)
                    for k, v in d.items()
                    if output_filter_func(k, v)
                },
            ]
            for d in [eval_pt_func() for _ in range(pool_size)]
        ]
    )


def run_module_with_pools(
    *,
    ait_module,
    inputs_pool,
    outputs_pool,
    num_iters,
    stream_ptr: Optional[int] = None,
    sync: bool = False,
    graph_mode: bool = False,
):
    """
    Run the module with the given inputs and outputs pools.
    Parameters
    ----------
    ait_module : Model
        The AIT module to run.
    inputs_pool : List[Dict[str, torch.Tensor]]
        A list of inputs to pass into Model.RunWithTensors.
    outputs_pool : List[Dict[str, torch.Tensor]]
        A list of outputs to pass into Model.RunWithTensors.
    num_iters : int
        The number of iterations to run.
    stream_ptr : Optional[int]
        The CUDA stream pointer to run the module on; if None, use the legacy stream.
    sync : bool
        Whether to synchronize the CUDA stream after each iteration.
    graph_mode : bool
        Whether to run the module in graph mode.
    """
    for i in range(num_iters):
        ait_module.run_with_tensors(
            inputs_pool[i % len(inputs_pool)],
            outputs_pool[i % len(outputs_pool)],
            sync=sync,
            stream_ptr=stream_ptr,
            graph_mode=graph_mode,
        )


def run_benchmark(
    *,
    ait_module,
    inputs_pool,
    outputs_pool,
    num_iters,
    num_warmup_iters,
    stream: Optional[torch.cuda.Stream] = None,
    sync: bool = False,
    graph_mode: bool = False,
):
    """
    Run the benchmark.
    Parameters
    ----------
    ait_module : Model
        The AIT module to run.
    inputs_pool : List[Dict[str, torch.Tensor]]
        A list of inputs to pass into Model.RunWithTensors.
    outputs_pool : List[Dict[str, torch.Tensor]]
        A list of outputs to pass into Model.RunWithTensors.
    num_iters : int
        The number of iterations to run.
    num_warmup_iters : int
        The number of warmup iterations to run.
    stream : Optional[torch.cuda.Stream]
        The CUDA stream to run the module on; if None, use the default stream.
    sync : bool
        Whether to synchronize the CUDA stream after each iteration.
    graph_mode : bool
        Whether to run the module in graph mode.
    Returns
    -------
    float
        The average time per iteration in *milliseconds*.
    """
    if stream is None:
        stream = torch.cuda.default_stream()

    _common_params = {
        "ait_module": ait_module,
        "inputs_pool": inputs_pool,
        "outputs_pool": outputs_pool,
        "sync": sync,
        "stream_ptr": stream.cuda_stream,
        "graph_mode": graph_mode,
    }
    # Warmup by running for num_warmup_iters
    run_module_with_pools(
        num_iters=num_warmup_iters,
        **_common_params,
    )
    # Benchmark by running for num_iters
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(stream=stream)
    run_module_with_pools(
        num_iters=num_iters,
        **_common_params,
    )
    end_event.record(stream=stream)
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters
