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
A subprocess based multiple GPUs runner for auto-tuning
"""
from __future__ import annotations

import concurrent
import os

import re
import subprocess
import typing
from collections import namedtuple
from queue import Queue
from typing import Callable, List, Tuple

from ..utils import logger
from .target import Target
from .task_runner import BaseRunner, Task

# pylint: disable=W0221

PROF_RUNTIME_PATTERN = re.compile(r"OP:([a-zA-Z0-9_]+),TIME:([\d\.]+),WS:([\d]+)")
# FIXME: We will remove the following two patterns once we implement the
# same profiling mechanism as gemm for conv and amd
RUNTIME_PATTERN = re.compile(r"TIME:([\d\.]+)")
WORKSPACE_PATTERN = re.compile(r"WS:([\d]+)")

ProfileResult = namedtuple("ProfileResult", "op_config duration workspace")
"""Object to store profiling result
"""


def optimization_key(result):
    return float(result[1])


def extract_profile_result(stdout) -> Tuple[ProfileResult, bool]:
    failed = False
    try:
        runtimes = PROF_RUNTIME_PATTERN.findall(stdout)
        if len(runtimes) > 0:
            logger.debug(__name__, f"all runtimes (unsorted): {runtimes}")
            # format - OP:xx,TIME:x.xx,WS:xx
            best_runtime = min(runtimes, key=optimization_key)
            op_config = best_runtime[0]
            duration = float(best_runtime[1])
            workspace = int(best_runtime[2])
        else:
            # FIXME: remove it once we unify our profiling mechanism for conv and amd
            op_config = ""
            duration = float(RUNTIME_PATTERN.findall(stdout)[0])
            workspace = int(WORKSPACE_PATTERN.findall(stdout)[0])
    except Exception:
        duration = 0
        workspace = 0
        failed = True
    return ProfileResult(op_config, duration, workspace), failed


def update_inplace(d, new_d):
    d.update(new_d)
    return d


def process_task(task: Task) -> None:
    """Extract kernel execution time and workspace from task process outputs

    Parameters
    ----------
    task : Task
        A profiling task
    """
    stdout = task._stdout
    stderr = task._stderr
    if len(stderr) > 0:
        logger.debug(
            __name__,
            "Failed: [{name}][{algo}]\ncmd:\n{cmd}\nstderr:\n{stderr}".format(
                name=task._name,
                algo=task._idx,
                cmd=task._cmd,
                stderr=stderr,
            ),
        )
    task._ret, task._failed = extract_profile_result(stdout)
    if not task._failed:
        logger.debug(
            __name__,
            f"Successful: [{task._name}][{task._idx}]: OP: {task._ret.op_config} "
            f"TIME: {task._ret.duration} WS:{task._ret.workspace}",
        )


def process_return(task: Task) -> typing.Tuple[typing.Union[int, str], ProfileResult]:
    """Generate profile result from a profiling task

    Parameters
    ----------
    task : Task
        A profiling task

    Returns
    -------
    Tuple[Union[int, str], ProfileResult]
        Tuple of task idx (usually the algorithm name/id) and profiling result
    """
    return (task._idx, task._ret)


class Runner(BaseRunner):
    """A parallel runner for multiple GPUs profiling tasks.
    Runner is inherited from BaseRunner.
    """

    def __init__(self, devs: list[int], op_name: str, timeout: int = 30):
        logger.info(
            __name__, "Using {n} GPU for profiling {op}".format(n=len(devs), op=op_name)
        )
        super().__init__(devs, op_name, timeout)
        self._dev_flag = Target.current().dev_select_flag()
        self._ftask_proc = process_task
        self._fret_proc = process_return

    def push(self, idx: typing.Union[int, str], cmd: str):
        """Push a new profiling task into runner's queue

        Parameters
        ----------
        idx : Union[int, str]
            Profiling task id (usually is algorithm id or name)
        cmd : str
            Bash command to execute the profiling task
        """
        self._queue.append(Task(idx, cmd, self._tag, dev_flag=self._dev_flag))

    def pull(self):
        """Pull results from all profiling tasks assigned to runner.

        Returns
        -------
        list[Tuple[Union[int, str], ProfileResult]]
            Profiling results of all successful tasks.
        """
        ret = super().pull(self._ftask_proc, self._fret_proc)
        return ret


def run_task(cmds, queue, dev_select_flag):
    # get device or block until one is available
    device = queue.get()
    logger.debug(__name__, f"running profiler {cmds=} on GPU #{device}")

    completed_process = subprocess.run(
        cmds,
        env=update_inplace(os.environ.copy(), {dev_select_flag: device}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    queue.put(device)
    return completed_process.stdout, completed_process.stderr


class ProfilerRunner:
    """Another parallel runner to execute profilers on multiple GPUs in parallel
    It uses a process pool for implementation, avoiding process creation overhead
    The size of the process pool is equal to the number of provided GPUs,
    so ~ideally~ each process should execute a profiler on its dedicated GPU.
    This property hasn't been properly verified yet,
    however, the results are empirically better compared to the previous runner.
    """

    def __init__(self, devices: List[str], timeout: int, postprocessing_delegate):
        """
        Parameters
        ----------
        devices : List[str]
            device identifiers (contents of {CUDA,HIP}_VISIBLE_DEVICES)
        timeout : int
            timeout to wait for all profilers completion in seconds
        postprocessing_delegate :
            object responsible for postprocessing results after futures completion
        """
        if devices is None:
            devices = [0]
        # This queue is used to ensure only one task is executed on a device at a time
        self._device_queue = Queue()
        # This queue is used to ensure postprocessing in `join()` happens *after* done_callbacks complete
        self._done_queue = Queue()
        for d in devices:
            self._device_queue.put(str(d))
        logger.info(__name__, f"Initialized profiler runner with devices: {devices}")
        self._timeout = timeout
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
        self._futures = []
        self._postprocessing_delegate = postprocessing_delegate
        self._dev_select_flag = Target.current().dev_select_flag()

    def push(self, cmds: List[str], process_result_callback: Callable):
        """
        Schedule the profiler for execution in a separate process,
        Call the callback after subprocess completion

        Parameters
        ----------
        cmds : List[str]
            argv for the launched profiler
        process_result_callback : Callable
            Called after subprocess completion in the main process
            (but possibly not main thread).
            Currently used to aggregate profiler results,
            so the callable takes `result` and `postprocessing_delegate` parameters
            It is also used to propagate the profiler launch context to the aggregation point,
            namely, split_k value for the gemm profilers
        """
        future = self._executor.submit(
            run_task, cmds, self._device_queue, self._dev_select_flag
        )

        # done callbacks are used to collect profiler results for postprocessing
        # they are launched asynchronously, in a separate thread,
        # some time after a future holding profiler result completes
        def callback_when_done(fut):
            try:
                stdout, stderr = fut.result()
                profile_result, err = extract_profile_result(stdout)
                if err:
                    logger.error(
                        f"Profiler failure!\nProfiler stdout: {stdout}\nProfiler stderr: {stderr}"
                    )
                    raise RuntimeError(f"Failed to extract profiler result for {cmds}")
                process_result_callback(profile_result, self._postprocessing_delegate)
            finally:
                # unblock one future in `join()`
                self._done_queue.put(stdout)

        future.add_done_callback(callback_when_done)
        self._futures.append(future)

    def join(self):
        """
        Wait for subprocesses completion or timeout; postprocess the profiler results with delegate(s)
        """
        done, not_done = concurrent.futures.wait(self._futures, self._timeout)
        for f in not_done:
            f.cancel()
        # block until each done_callback completes,
        # or raise Empty exception after 3 minutes of waiting
        block_timeout = 3 * 60
        for _ in self._futures:
            self._done_queue.get(timeout=block_timeout)
        self._postprocessing_delegate.postprocess_results()
