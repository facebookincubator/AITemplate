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

import re
import typing
from collections import namedtuple

from ..utils import logger
from .target import Target
from .task_runner import BaseRunner, Task

# pylint: disable=W0221

RUNTIME_PATTERN = re.compile(r"TIME:([\d\.]+)")
WORKSPACE_PATTERN = re.compile(r"WS:([\d]+)")

ProfileResult = namedtuple("ProfileResult", "duration workspace")
"""Object to store profiling result
"""


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
        task._failed = True
        logger.debug(
            __name__,
            "Failed: [{name}][{algo}]\ncmd:\n{cmd}\nstderr:\n{stderr}".format(
                name=task._name,
                algo=task._idx,
                cmd=task._cmd,
                stderr=stderr,
            ),
        )
    else:
        duration = float(RUNTIME_PATTERN.findall(stdout)[0])
        workspace = int(WORKSPACE_PATTERN.findall(stdout)[0])
        task._ret = ProfileResult(duration, workspace)
        logger.info(
            __name__,
            "Successful: [{name}][{algo}]: TIME: {duration} WS:{ws}".format(
                name=task._name, algo=task._idx, duration=duration, ws=workspace
            ),
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
