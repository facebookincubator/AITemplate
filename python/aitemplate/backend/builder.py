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

import logging
import multiprocessing

import os
import re
import shlex
import subprocess
import typing
from hashlib import sha1
from pathlib import Path
from typing import Optional

import jinja2

from aitemplate.utils.debug_settings import AITDebugSettings

from ..utils.misc import is_debug
from .target import Target
from .task_runner import BaseRunner, Task

# pylint: disable=W0221,C0103


_LOGGER = logging.getLogger(__name__)
_DEBUG_SETTINGS = AITDebugSettings()


def _augment_for_trace(cmd):
    return (
        'date +"{{\\"name\\": \\"$@\\", \\"ph\\": \\"B\\", \\"pid\\": \\"$$$$\\", \\"ts\\": \\"%s%6N\\"}},";'
        " {}; "
        'date +"{{\\"name\\": \\"$@\\", \\"ph\\": \\"E\\", \\"pid\\": \\"$$$$\\", \\"ts\\": \\"%s%6N\\"}},";'
    ).format(cmd)


def _time_cmd(cmd):
    return f"exec time -f 'exit_status=%x elapsed_sec=%e argv=\"%C\"' {cmd}"


def _log_error_context(
    stderr,
    build_dir,
    context_radius=10,
    max_errors_per_file=5,
    padding=5,
):
    path_to_error_lines = {}
    for line in [L for L in stderr.split("\n") if ": error:" in L]:
        match = re.search(r"(.+)\((\d+)\): error:.*", line)
        if match:
            path = match[1]
            error_line = match[2]
            if path not in path_to_error_lines:
                path_to_error_lines[path] = set()
            # nvcc line numbers are 1-based
            error_line = int(error_line) - 1
            path_to_error_lines[path].add(error_line)

    # keep only the first N error lines per file
    path_to_error_lines = {
        path: sorted(error_lines)[:max_errors_per_file]
        for path, error_lines in path_to_error_lines.items()
    }

    path_to_visible_lines = {}
    for path, error_lines in path_to_error_lines.items():
        path_to_visible_lines[path] = set()
        for error_line in error_lines:
            # collect the context lines around each error line
            context = range(
                error_line - context_radius,
                error_line + context_radius + 1,
            )
            path_to_visible_lines[path].update(list(context))

    for path, visible_lines in path_to_visible_lines.items():
        full_path = os.path.join(build_dir, path)
        if os.path.exists(full_path):
            # read the lines from the file
            with open(full_path, "r") as f:
                # each line ends with '\n'
                file_lines = f.readlines()
            # except maybe the last line
            if file_lines and not file_lines[-1].endswith("\n"):
                file_lines[-1] = f"{file_lines[-1]}\n"
            num_file_lines = len(file_lines)

            error_lines = path_to_error_lines[path]
            visible_lines = sorted(visible_lines)

            lines_to_show = []
            last_printed_i = -1
            for i in visible_lines:
                if i < 0 or i >= num_file_lines:
                    # skip the line number as extraneous
                    continue
                if i - last_printed_i > 1:
                    # preceding ellipsis
                    lines_to_show.append("...\n")
                line = file_lines[i]
                lines_to_show.append(f"{i+1:<{padding}} {line}")
                if i in error_lines:
                    # mark the line as an error line: underscore
                    spaces = line[: len(line) - len(line.lstrip())]
                    underscore = spaces + "^" * (len(line) - len(spaces) - 1)
                    lines_to_show.append(f"{' ' * padding} {underscore}\n")
                last_printed_i = i
            if visible_lines[-1] < num_file_lines - 1:
                # closing ellipsis
                lines_to_show.append("...\n")

            if lines_to_show:
                # all lines_to_show end with '\n'
                summary = "".join(lines_to_show)
                _LOGGER.info(f"{path}:\n\n{summary}")


def _run_make_cmds(cmds, timeout, build_dir):
    _LOGGER.debug(f"make {cmds=}")
    proc = subprocess.Popen(
        [" && ".join(cmds)],
        shell=True,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(timeout)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        out, err = proc.communicate()
        raise e
    finally:
        stdout = out.decode()
        stderr = err.decode()
        if proc.returncode != 0:
            _LOGGER.info(f"make stdout:\n\n{stdout}")
            _LOGGER.info(f"make stderr:\n\n{stderr}")

            _log_error_context(stderr, build_dir)

            raise RuntimeError("Build has failed.")
        else:
            _LOGGER.debug(f"make stdout:\n\n{stdout}")
            _LOGGER.debug(f"make stderr:\n\n{stderr}")


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
        _LOGGER.info(
            "Failed: [{name}]\ncmd:\n{cmd}\nstderr:\n{stderr}\nstdout:{stdout}".format(
                name=task._name, cmd=task._cmd, stderr=stderr, stdout=stdout
            ),
        )
        task._ret = -1
    else:
        _LOGGER.debug(
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
        _LOGGER.info(
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
        self._n_jobs = n_jobs
        self._timeout = timeout
        self._do_trace = os.environ.get("AIT_TRACE_MAKE", False)

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
            _LOGGER.info("Building " + target)
            if src.endswith(".bin"):
                if binary_cc_cmd is None:
                    raise ValueError(
                        "Cannot compile .bin file without specifying binary_cc_cmd!"
                    )

                src_path = Path(src)
                target_path = Path(target)
                compile_cmd = binary_cc_cmd.format(
                    target=target_path.name, src=src_path.name
                )
                containing_dir = str(src_path.parent.absolute())
                # Have to cd into the containing dir so ld doesn't include
                # the path in the symbol names; unfortunately, there's no other
                # way to control this.
                if is_debug():
                    cmd = f"cd {containing_dir} && {compile_cmd} && cd -"
                else:
                    # If not in debug mode, remove the original .bin file which can potentially be quite large.
                    cmd = f"cd {containing_dir} && {compile_cmd} && rm {src_path.name} && cd -"
            else:
                cmd = cc_cmd.format(target=target, src=src)

            cmd = _time_cmd(cmd)
            _LOGGER.debug(f"The cmd for building {target} is : {cmd}")
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
        _LOGGER.info("Building " + target)
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
        cmd = _time_cmd(cmd)
        _LOGGER.debug(f"The cmd for building {target} is {cmd}")
        self._runner.push(0, cmd, target)
        self._runner.join()
        self._runner.pull()

    def gen_makefile(self, file_pairs, dll_name, workdir, test_name, debug_settings):

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

.PHONY: all clean clean_constants
all: {{targets}}

{{dll_target}}: $(obj_files)
    {{build_so_cmd}}

{{build_standalone_rules}}

clean:
    rm -f *.obj {{targets}}

clean_constants:
    rm -f constants.bin
"""
        )

        standalone_rules_template = jinja2.Template(
            """
{{standalone_src}}: {{standalone_obj}}
    {{cfile_cmd}}

{{exe_target}}: {{exe_target_deps}}
    {{build_exe_cmd}}
"""
        )

        build_so_cmd = "$(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files)"
        standalone_src = "standalone.cu"
        standalone_obj = "standalone.obj"
        obj_files = []
        # standalone.cu is an AITemplate internal file that is used for generating
        # standalone executables. We only want to compile it when the relevant
        # debug option is enabled.
        obj_files = [
            pair[1].split("/")[-1]
            for pair in file_pairs
            if not pair[1].endswith(standalone_obj)
        ]
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

        if self._do_trace:
            cfile_cmd = _augment_for_trace(cfile_cmd)
            bfile_cmd = _augment_for_trace(bfile_cmd)
            build_so_cmd = _augment_for_trace(build_so_cmd)
        else:
            cfile_cmd = _time_cmd(cfile_cmd)
            bfile_cmd = _time_cmd(bfile_cmd)
            build_so_cmd = _time_cmd(build_so_cmd)

        build_exe_cmd = _time_cmd("$(CC) $(CFLAGS) -o $@ $(obj_files)")
        targets = f"{dll_name}"

        build_standalone_rules = ""
        if debug_settings.gen_standalone:
            build_exe_cmd = f"$(CC) $(CFLAGS) -o $@ {standalone_obj} {dll_name}"
            exe_name = os.path.splitext(dll_name)[0] + ".exe"
            exe_target_deps = f"{dll_name} {standalone_obj}"
            build_standalone_rules = standalone_rules_template.render(
                standalone_src=standalone_src,
                standalone_obj=standalone_obj,
                cfile_cmd=cfile_cmd,
                exe_target=exe_name,
                exe_target_deps=exe_target_deps,
                build_exe_cmd=build_exe_cmd,
            )
            targets += f" {exe_name}"

        makefile_str = makefile_template.render(
            cc=cc,
            cpp=cpp,
            CFLAGS=compile_options,
            fPIC=fpic,
            obj_files=obj_files,
            dll_target=dll_name,
            targets=targets,
            cfile_cmd=cfile_cmd,
            bfile_cmd=bfile_cmd,
            build_so_cmd=build_so_cmd,
            build_standalone_rules=build_standalone_rules,
        )

        dumpfile = os.path.join(workdir, test_name, "Makefile")
        with open(dumpfile, "w+") as f:
            # fix the makefile indentation
            f.write(re.sub("^    ", "\t", makefile_str, flags=re.M))

    @staticmethod
    def _combine_profiler_multi_sources():
        """Whether to combine multiple profiler sources per target."""
        return bool(int(os.environ.get("COMBINE_PROFILER_MULTI_SOURCES", 1)))

    @staticmethod
    def _force_one_profiler_source_per_target():
        """Whether to combine multiple profiler sources per target into one."""
        return bool(int(os.environ.get("FORCE_ONE_PROFILER_SOURCE_PER_TARGET", 0)))

    def _combine_sources(self, sources):
        """
        Combine multiple source files (given by path) into one
        source file and return the path of the combined file.

        Parameters
        ----------
        sources : Iterable[str]
            The list of paths to the source files to combine.

        Returns
        -------
        path : str
            The path to the combined source file.
        """
        assert len(sources) > 0, "Must have at least one source"
        if len(sources) == 1:
            # no need to combine a single source
            return next(iter(sources))

        file_lines = []
        for source in sources:
            with open(source, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip():
                    # collect the original non-empty lines
                    file_lines.append(line)
            # the last line might not end with "\n"
            file_lines.append("\n")

        # generate a new file name conditioned on the list of the source file names
        file_name = sha1((";".join(sorted(sources))).encode("utf-8")).hexdigest()
        file_dir = Path(next(iter(sources))).parents[0]  # fetch the directory
        file_path = file_dir / Path(f"temp_{file_name}.cu")
        with open(file_path, "w") as f:
            # file_lines end with "\n" already
            f.write("".join(file_lines))

        # return the path starting with "./"
        return os.path.join(".", str(file_path))

    def _combine_profiler_sources(self, target_to_sources, num_builders):
        """
        Combine multiple profiler sources generated for different targets
        to optimize the overall compilation time, given the available number
        of builders (CPUs). The total number of sources (across all targets)
        is set equal to the `num_builders`. Single-source targets are kept
        as is; multi-source targetss' sources are possibly combined.

        Simplifying assumptions:

            - Individual split (multiple) sources per target take
              approximately equal time to compile across different
              targets (this is, in particular, not true for the main
              profiler source file vs kernel-specific source files:
              the former is typically larger than the latter);
            - Compilation time grows linearly in the number of
              separate sources combined into a single file.

        Parameters
        ----------
        target_to_soruces : dict[str, Iterable[str]]
            The mapping from each target name to the list of sources
            required to compile this target. There can be one or more
            sources for each target.
        num_builders : int
            The number of available builders (CPUs).

        Returns
        ----------
        target_to_combined_sources : dict[str, Iterable[str]]
            Like `target_to_sources`, but with some of the source paths
            in the values replaced by the paths to the respective combined
            source files. Whether and which of the sources are combined
            depends on the arguments.
        """
        num_total_sources = num_builders

        if (
            len(target_to_sources) >= num_total_sources
            or self._force_one_profiler_source_per_target()
        ):
            # there are at least as many targets as the total
            # number of sources required (or single source per
            # target is forced): combine everything
            return {
                target: [self._combine_sources(sources)]
                for target, sources in target_to_sources.items()
            }

        combine_candidates = {}  # multi-source targets
        num_multi_sources, num_single_sources = 0, 0
        for target, sources in target_to_sources.items():
            if len(sources) > 1:
                combine_candidates[target] = sources
                num_multi_sources += len(sources)
            else:
                num_single_sources += 1

        if num_multi_sources == 0:
            # all targets are single-source: nothing to combine
            return target_to_sources
        if num_multi_sources + num_single_sources <= num_total_sources:
            # there are fewer source files than the total
            # number of sources required: no need to combine
            return target_to_sources

        # number of sources we need for the multi-file targets
        num_combined_sources = num_total_sources - num_single_sources
        num_sources_per_target = {
            # the number of combined sources per multi-source target as a
            # fraction of num_combined_sources is proportional to the number of
            # multiple sources of the target (rounded down); ultimately, there
            # should be at least one source target (hence max(..., 1))
            target: max(int(len(sources) / num_multi_sources * num_combined_sources), 1)
            for target, sources in combine_candidates.items()
        }

        # do any sources remain after the above per-target distribution?
        remaining_sources = num_combined_sources - sum(num_sources_per_target.values())
        if remaining_sources > 0:
            # reverse-sort the targets by the remainder after rounding down:
            # prefer adding sources to the targets with a higher remainder
            # (i.e. the ones closest to getting another source)
            targets = sorted(
                num_sources_per_target.keys(),
                key=lambda target: (
                    (
                        len(target_to_sources[target])
                        / num_multi_sources
                        * num_combined_sources
                    )
                    - int(
                        len(target_to_sources[target])
                        / num_multi_sources
                        * num_combined_sources
                    )
                ),
                reverse=True,
            )
            target_id = 0
            while remaining_sources > 0:
                # increment the number of sources for the target
                num_sources_per_target[targets[target_id]] += 1
                target_id = (target_id + 1) % len(targets)
                remaining_sources -= 1

        result = {}
        for target in target_to_sources:
            if target in combine_candidates:
                # collect the sources of the target
                # in N batches by round robin
                num_sources = num_sources_per_target[target]
                # TODO: form the source batches by the total number
                # of lines instead of the number of sources for more
                # even distribution of the compilation time per batch
                batch_id = 0
                batches = [[] for _ in range(num_sources)]
                for source in target_to_sources[target]:
                    batches[batch_id].append(source)
                    batch_id = (batch_id + 1) % num_sources
                # conbine the sources in each batch
                result[target] = [self._combine_sources(b) for b in batches]
            else:
                # use the single-source profiler target as is
                result[target] = target_to_sources[target]
        return result

    def _gen_makefile_for_profilers(self, file_pairs, profiler_dir):
        makefile_template = jinja2.Template(
            """
all: {{targets}}
.PHONY: all clean

{{commands}}

clean:
\trm -f {{targets}}
"""
        )
        # normalize the profiler dir: add / at the end
        profiler_dir = os.path.join(profiler_dir, "")

        # deduplicate targets from different ops
        target_to_sources = {}
        for source, target in file_pairs:
            if target not in target_to_sources:
                target_to_sources[target] = set()
            if isinstance(source, str):
                target_to_sources[target].add(source)
            else:
                target_to_sources[target].update(source)

        # stabilize the order of sources per target
        target_to_sources = {
            target: sorted(sources) for target, sources in target_to_sources.items()
        }

        if self._combine_profiler_multi_sources():
            num_sources_before = sum(len(s) for s in target_to_sources.values())
            target_to_sources = self._combine_profiler_sources(
                target_to_sources=target_to_sources,
                num_builders=self._n_jobs,
            )
            num_sources_after = sum(len(s) for s in target_to_sources.values())

            _LOGGER.info(
                f"combined {num_sources_before} profiler sources into {num_sources_after}",
            )

        targets = []
        dependencies = {}
        for target, sources in target_to_sources.items():
            target = target.split(profiler_dir)[-1]
            if len(sources) == 1:
                # single-source profiler executable
                source = next(iter(sources))
                source = source.split(profiler_dir)[-1]
                dependencies[target] = [source]
            else:
                # multi-source profiler executable
                objects = []
                for source in sources:
                    # first compile the objects
                    source = source.split(profiler_dir)[-1]
                    obj = source.replace(".cu", ".obj")
                    if not os.path.exists(os.path.join(profiler_dir, obj)):
                        # compile the object only if it is absent
                        dependencies[obj] = [source]
                    objects.append(obj)
                # then link the objects into an executable
                dependencies[target] = objects
            targets.append(target)

        commands = []
        num_compiled_sources = 0
        num_linked_executables = 0
        for target, srcs in dependencies.items():
            # for each "target: srcs" pair,
            # generate two lines for the Makefile
            src_list = " ".join(srcs)
            dep_line = f"{target}: {src_list}"
            cmd_line = (
                Target.current()
                .compile_cmd(executable=(not target.endswith(".obj")))
                .format(target=target, src=src_list)
            )
            if self._do_trace:
                cmd_line = _augment_for_trace(cmd_line)
            else:
                cmd_line = _time_cmd(cmd_line)

            command = f"{dep_line}\n\t{cmd_line}\n"
            commands.append(command)

            # increment compilation statistics
            num_compiled_sources += sum(1 for s in srcs if s.endswith(".cu"))
            num_linked_executables += 0 if target.endswith(".obj") else 1

        _LOGGER.info(f"compiling {num_compiled_sources} profiler sources")
        _LOGGER.info(f"linking {num_linked_executables} profiler executables")

        makefile_str = makefile_template.render(
            targets=" ".join(set(targets)),
            commands="\n".join(commands),
        )

        dumpfile = os.path.join(profiler_dir, "Makefile")
        with open(dumpfile, "w+") as f:
            f.write(makefile_str)

    def make_profilers(self, generated_profilers, workdir):
        file_pairs = [f for gp in generated_profilers for f in gp]
        if not file_pairs:
            return
        build_dir = shlex.quote(os.path.join(workdir, "profiler"))
        self._gen_makefile_for_profilers(file_pairs, build_dir)
        make_path = shlex.quote(Target.current().make())
        make_flags = " ".join(
            [
                "--output-sync",
                f"-C {build_dir}",
            ]
        )
        make_clean_cmd = f" {make_path} {make_flags} clean "
        make_all_cmd = f" {make_path} {make_flags} -j{self._n_jobs} all "
        cmds = [make_clean_cmd, make_all_cmd]
        _run_make_cmds(cmds, self._timeout, build_dir)

    def make(
        self, file_pairs, dll_name, workdir, test_name, debug_settings=_DEBUG_SETTINGS
    ):
        self.gen_makefile(file_pairs, dll_name, workdir, test_name, debug_settings)
        make_path = shlex.quote(Target.current().make())
        build_dir = shlex.quote(os.path.join(workdir, test_name))
        make_flags = " ".join(
            [
                "--output-sync",
                f"-C {build_dir}",
            ]
        )
        make_clean_cmd = f" {make_path} {make_flags} clean "
        make_all_cmd = f" {make_path} {make_flags} -j{self._n_jobs} all "
        make_clean_constants_cmd = f" {make_path} {make_flags} clean_constants "
        cmds = [make_clean_cmd, make_all_cmd]
        if not is_debug():
            cmds.append(make_clean_constants_cmd)
        _run_make_cmds(cmds, self._timeout, build_dir)
