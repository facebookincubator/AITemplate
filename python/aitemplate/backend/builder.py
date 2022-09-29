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
Builder is a module to compile generated source code files into binary objects.
"""

from __future__ import annotations

import multiprocessing

import os
import pathlib
import re
import typing
from typing import Optional

import jinja2

from ..utils import logger
from .target import Target
from .task_runner import BaseRunner, Task

# pylint: disable=W0221,C0103


def process_task(task: Task) -> None:
    """This function extracts stdout and stderr from a finished task.
    If the task process return code is not 0, will mark the task as
    a failed task.

    Parameters
    ----------
    task : Task
        A compiling task
    """
    stdout = task._stdout
    stderr = task._stderr
    if task._proc.returncode != 0:
        task._failed = True
        logger.info(
            __name__,
            "Failed: [{name}]\ncmd:\n{cmd}\nstderr:\n{stderr}\nstdout:{stdout}".format(
                name=task._name, cmd=task._cmd, stderr=stderr, stdout=stdout
            ),
        )
        task._ret = -1
    else:
        logger.debug(
            __name__,
            "Successful: [{name}]\ncmd:\n{cmd}\nstderr:\n{stderr}\nstdout:{stdout}".format(
                name=task._name, cmd=task._cmd, stderr=stderr, stdout=stdout
            ),
        )
        task._ret = 0


def process_return(task: Task) -> None:
    """This function process the task. If task is timeout or failed,
    raise a runtime error.

    Parameters
    ----------
    task : Task
        A compiling task.

    Raises
    ------
    RuntimeError
        Compiling failed.
    """
    if not task.is_timeout() and task.is_failed():
        raise RuntimeError(f"Building failed. Logs:\n{task._stdout}\n{task._stderr}")


class Runner(BaseRunner):
    """A parallel runner for compiling tasks.
    Runner is inherited from BaseRunner.
    """

    def __init__(self, devs: list[int], timeout: int = 10):
        """Initialize a parallel runner for building

        Parameters
        ----------
        devs : list[int]
            CPU ids for compiling
        timeout : int, optional
            Compiling timeout, by default 10 (seconds)
        """
        super().__init__(devs, "builder", timeout)
        logger.info(
            __name__,
            "Using {n} CPU for building".format(n=devs),
        )
        self._ftask_proc = process_task
        self._fret_proc = process_return

    def push(self, idx: typing.Union[int, str], cmd: str, target: Target) -> None:
        """Push a building task into runner

        Parameters
        ----------
        idx : Union[int, str]
            Task id
        cmd : str
            bash command for compiling
        target : Target
            Target device type for building
        """
        self._queue.append(Task(idx, cmd, target, shell=True))

    def pull(self) -> list[None]:
        """Pull building results.
        Check whether all building tasks are successful.

        Returns
        -------
        list
            An empty list
        """
        ret = super().pull(self._ftask_proc, self._fret_proc)
        return ret


class Builder(object):
    """Builder is a module to compile generated source code
    files into binary objects.
    """

    def __init__(self, n_jobs: int = -1, timeout: int = 180) -> None:
        """Initialize a parallel builder for compiling source code.

        Parameters
        ----------
        n_jobs : int, optional
            Run how many parallel compiling job,
            by default -1, which will set n_jobs to `multiprocessing.cpu_count()`
        timeout : int, optional
            Timeout value, by default 180 (seconds)
        """
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()
        num_builder = os.environ.get("NUM_BUILDERS", None)
        if num_builder is not None:
            n_jobs = int(num_builder)
        self._runner = Runner(n_jobs, timeout)

    def build_objs(
        self,
        files: list[typing.Tuple[str, str]],
        cc_cmd: str,
        binary_cc_cmd: Optional[str] = None,
    ):
        """Generate building task for each source code file, then build in parallel

        Parameters
        ----------
        files : list[Tuple[str, str]]
            list of tuples of source code path and object file path
        cc_cmd : str
            command line template for building objects
        binary_cc_cmd : optional, str
            command line template for turning raw binary files (those ending in .bin) into
            objects. Since most compilation jobs will not need to compile these, this argument
            is optional.
        """
        for idx, fpair in enumerate(files):
            src, target = fpair
            logger.info(__name__, "Building " + target)
            if src.endswith(".bin"):
                if binary_cc_cmd is None:
                    raise ValueError(
                        "Cannot compile .bin file without specifying binary_cc_cmd!"
                    )

                src_path = pathlib.Path(src)
                target_path = pathlib.Path(target)
                compile_cmd = binary_cc_cmd.format(
                    target=target_path.name, src=src_path.name
                )
                containing_dir = str(src_path.parent.absolute())
                # Have to cd into the containing dir so ld doesn't include
                # the path in the symbol names; unfortunately, there's no other
                # way to control this.
                if logger.is_debug():
                    cmd = f"cd {containing_dir} && {compile_cmd} && cd -"
                else:
                    # If not in debug mode, remove the original .bin file which can potentially be quite large.
                    cmd = f"cd {containing_dir} && {compile_cmd} && rm {src_path.name} && cd -"
            else:
                cmd = cc_cmd.format(target=target, src=src)

            logger.debug(__name__, f"The cmd for building {target} is : {cmd}")
            self._runner.push(idx, cmd, target)
        self._runner.join()
        self._runner.pull()

    def build_so(self, target: Target, objs: list[str]):
        """Generate a task to build all objects into a dynamic library

        Parameters
        ----------
        target : Target
            Device target of dynamic library
        objs : list[str]
            List of all object file paths for building the dynamic library.
        """
        logger.info(__name__, "Building " + target)
        cc = Target.current().cc()
        compile_options = Target.current().compile_options()
        fpic = "-fPIC"
        if "nvcc" in cc:
            fpic = "-Xcompiler=-fPIC"
        cmd = (
            "{cc} -shared ".format(cc=cc)
            + fpic
            + " "
            + compile_options
            + " -o {target} {objs}".format(target=target, objs=" ".join(objs))
        )
        logger.debug(__name__, f"The cmd for building {target} is {cmd}")
        self._runner.push(0, cmd, target)
        self._runner.join()
        self._runner.pull()

    def gen_makefile(self, file_pairs, dll_name, workdir, test_name):

        makefile_template = jinja2.Template(
            """
CC = {{cc}}
CFLAGS = {{CFLAGS}}
fPIC_flag = {{fPIC}}

obj_files = {{obj_files}}

%.obj : %.{{cpp}}
    {{cfile_cmd}}
%.obj : %.bin
    {{bfile_cmd}}

.PHONY: all
all: {{target}}

{{target}}: $(obj_files)
    $(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files)

clean:
    rm -f *.obj test.so
"""
        )

        obj_files = [pair[1].split("/")[-1] for pair in file_pairs]
        obj_files = " ".join(obj_files)

        cc = Target.current().cc()
        compile_options = Target.current().compile_options()

        fpic, cpp = "-fPIC", "cpp"
        if "nvcc" in cc:
            fpic, cpp = "-Xcompiler=-fPIC", "cu"

        cfile_cmd = Target.current().compile_cmd(False).format(target="$@", src="$<")
        bfile_cmd = Target.current().binary_compile_cmd()
        if not bfile_cmd:
            bfile_cmd = ""
        else:
            bfile_cmd = bfile_cmd.format(target="$@", src="$<")

        makefile_str = makefile_template.render(
            cc=cc,
            cpp=cpp,
            CFLAGS=compile_options,
            fPIC=fpic,
            obj_files=obj_files,
            target=dll_name,
            cfile_cmd=cfile_cmd,
            bfile_cmd=bfile_cmd,
        )

        dumpfile = os.path.join(workdir, test_name, "Makefile")
        with open(dumpfile, "w+") as f:
            # fix the makefile indentation
            f.write(re.sub("^    ", "\t", makefile_str, flags=re.M))
