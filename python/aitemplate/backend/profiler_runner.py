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

import concurrent.futures
import logging
import os

import re
import subprocess
from collections import namedtuple
from queue import Queue
from time import sleep
from typing import Callable, List, Tuple, Union

from aitemplate.backend.target import Target
from aitemplate.backend.task_runner import BaseRunner, Task

from aitemplate.testing import detect_target

# pylint: disable=W0221


_LOGGER = logging.getLogger(__name__)


PROF_RUNTIME_PATTERN = re.compile(r"OP:([a-zA-Z0-9_]+),TIME:([\d\.]+),WS:([\d]+)")
# FIXME: We will remove the following two patterns once we implement the
# same profiling mechanism as gemm for conv and amd
RUNTIME_PATTERN = re.compile(r"TIME:([\d\.]+)")
WORKSPACE_PATTERN = re.compile(r"WS:([\d]+)")

PROFILER_RUN_MAX_ATTEMPTS = 3
PROFILER_RUN_RETRY_DELAY_SECONDS = 5

ProfileResult = namedtuple("ProfileResult", "op_config duration workspace")
"""Object to store profiling result
"""


def optimization_key(result):
    return float(result[1])


def extract_profile_result(
    stdout,
    return_ops=None,
) -> Tuple[Union[ProfileResult, List[ProfileResult]], bool]:
    failed = False
    try:
        runtimes = PROF_RUNTIME_PATTERN.findall(stdout)
        if len(runtimes) > 0:
            _LOGGER.debug(f"all runtimes (unsorted): {runtimes}")
            # format - OP:xx,TIME:x.xx,WS:xx
            if return_ops is not None:
                _LOGGER.debug(f"return ops: {return_ops}")
                return_ops = set(return_ops)
                result = [
                    ProfileResult(
                        op_config=runtime[0],
                        duration=float(runtime[1]),
                        workspace=int(runtime[2]),
                    )
                    for runtime in runtimes
                    if runtime[0] in return_ops
                ]
            else:
                best_runtime = min(runtimes, key=optimization_key)
                result = ProfileResult(
                    op_config=best_runtime[0],
                    duration=float(best_runtime[1]),
                    workspace=int(best_runtime[2]),
                )
        else:
            # FIXME: remove it once we unify our profiling mechanism for conv and amd
            result = ProfileResult(
                op_config="",
                duration=float(RUNTIME_PATTERN.findall(stdout)[0]),
                workspace=int(WORKSPACE_PATTERN.findall(stdout)[0]),
            )
    except Exception:
        result = ProfileResult(
            op_config="",
            duration=float("inf"),
            workspace=0,
        )
        failed = True
    return result, failed


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
    single_file_profiler = False

    if len(stderr) > 0:
        # TODO: ugly fix, should remove when finish all profiler refactor
        _LOGGER.debug(
            "Failed: [{name}][{algo}]\ncmd:\n{cmd}\nstderr:\n{stderr}".format(
                name=task._name,
                algo=task._idx,
                cmd=task._cmd,
                stderr=stderr,
            ),
        )
        runtimes = PROF_RUNTIME_PATTERN.findall(stdout)
        if len(runtimes) > 0:
            single_file_profiler = True
        if not single_file_profiler:
            task._failed = True
            return

    task._ret, task._failed = extract_profile_result(
        stdout=stdout,
        return_ops=task._kwargs.get("return_ops", None),
    )
    if not task._failed:
        results = task._ret
        if not isinstance(results, list):
            results = [results]
        for result in results:
            _LOGGER.debug(
                f"Successful: [{task._name}][{task._idx}]: OP: {result.op_config} "
                f"TIME: {result.duration} WS:{result.workspace}",
            )


def process_return(task: Task) -> Tuple[Union[int, str], ProfileResult]:
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

    def __init__(self, devs: List[int], op_name: str, timeout: int = 30):
        _LOGGER.info("Using {n} GPU for profiling {op}".format(n=len(devs), op=op_name))
        super().__init__(devs, op_name, timeout)
        self._dev_flag = Target.current().dev_select_flag()
        self._ftask_proc = process_task
        self._fret_proc = process_return

    def push(self, idx: Union[int, str], cmd: str, return_ops: List[str] = None):
        """Push a new profiling task into runner's queue

        Parameters
        ----------
        idx : Union[int, str]
            Profiling task id (usually is algorithm id or name)
        cmd : str
            Bash command to execute the profiling task
        return_ops : List[str]
            Names of the ops to return the profiling results for. If specified,
            instead of a single (best) ProfileResult instance, a list with the
            ProfileResults for each op in the return_ops is returned from `pull`.
        """
        self._queue.append(
            Task(
                idx,
                cmd,
                self._tag,
                dev_flag=self._dev_flag,
                return_ops=return_ops,
            )
        )

    def pull(self):
        """Pull results from all profiling tasks assigned to runner.

        Returns
        -------
        List[Tuple[Union[int, str], ProfileResult]]
            Profiling results of all successful tasks.
        """
        ret = super().pull(self._ftask_proc, self._fret_proc)
        return ret


def run_task(cmds, queue, dev_select_flag):
    # get device or block until one is available
    device = queue.get()
    _LOGGER.debug(f"running profiler {cmds=} on GPU #{device}")

    attempts = 0
    while True:
        try:
            completed_process = subprocess.run(
                cmds,
                env=update_inplace(os.environ.copy(), {dev_select_flag: device}),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            break
        except Exception as ex:
            attempts += 1
            if attempts >= PROFILER_RUN_MAX_ATTEMPTS:
                raise
            _LOGGER.debug(
                f"[{attempts} / {PROFILER_RUN_MAX_ATTEMPTS}] "
                f"Failed to run profiler {cmds=} due to exception: {ex}. "
                f"Will retry in {PROFILER_RUN_RETRY_DELAY_SECONDS} seconds."
            )
            sleep(PROFILER_RUN_RETRY_DELAY_SECONDS)

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

    def __init__(self, devices: List[str], postprocessing_delegate, timeout: int = 500):
        """
        Parameters
        ----------
        devices : List[str]
            device identifiers (contents of {CUDA,HIP}_VISIBLE_DEVICES)
        postprocessing_delegate :
            object responsible for postprocessing results after futures completion
        timeout : int
            timeout to wait for all profilers completion in seconds
        """
        if not devices:
            # devices is either None or empty list: use device 0
            devices = [0]
        # This queue is used to ensure only one task is executed on a device at a time
        self._device_queue = Queue()
        # This queue is used to ensure postprocessing in `join()` happens *after* done_callbacks complete
        self._done_queue = Queue()
        for d in devices:
            self._device_queue.put(str(d))
        _LOGGER.info(f"Initialized profiler runner with devices: {devices}")
        self._timeout = timeout
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
        self._futures = []
        self._cmds = []
        self._postprocessing_delegate = postprocessing_delegate
        try:
            target = Target.current()
        except RuntimeError:
            target = detect_target()
        self._dev_select_flag = target.dev_select_flag()

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
            stdout = None
            stderr = None
            try:
                stdout, stderr = "", ""
                stdout, stderr = fut.result()
                profile_result, err = extract_profile_result(stdout)
                if err:
                    _LOGGER.debug(
                        f"Profiler failure!\nProfiler stdout: {stdout}\nProfiler stderr: {stderr}",
                    )
                    raise RuntimeError(f"Failed to extract profiler result for {cmds}")
                process_result_callback(profile_result, self._postprocessing_delegate)
            finally:
                # unblock one future in `join()`
                if stdout is not None:
                    self._done_queue.put(stdout)

        future.add_done_callback(callback_when_done)
        self._futures.append(future)
        self._cmds.append(cmds)

    def join(self):
        """
        Wait for subprocesses completion or timeout; postprocess the profiler results with delegate(s)
        """
        done, not_done = concurrent.futures.wait(self._futures, self._timeout)
        for f in not_done:
            # attempts cancelling, will fail if call is being executed or has finished
            f.cancel()
        cancelled_cmds = [
            cmd for cmd, f in zip(self._cmds, self._futures) if f.cancelled()
        ]
        if cancelled_cmds:
            raise RuntimeError(
                f"Profiler timed out after {self._timeout} sec. "
                "Try increasing the timeout. "
                f"Cancelled profilers: {cancelled_cmds}"
            )
        for _ in [f for f in self._futures if f.done() or f.running()]:
            # sync point between futures and queue.
            # wait for callbacks to finish
            self._done_queue.get(timeout=self._timeout)
        self._postprocessing_delegate.postprocess_results()
