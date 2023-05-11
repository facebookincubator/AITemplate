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

# A custom compile engine for CMake for CUDA backend. It can handle both Windows
# and Linux use cases. Unlike the default make-based compiler engine, this one
# is an experimental one. It was mostly needed to generate cpp/cu files for a
# given model once and then do some custom debugging / research in an IDE.

from __future__ import annotations

import logging

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import jinja2

from aitemplate.backend.target import Target

from aitemplate.utils.debug_settings import AITDebugSettings

from aitemplate.utils.misc import is_linux, is_windows, short_str


# pylint: disable=W0221,C0103


_LOGGER = logging.getLogger(__name__)
_DEBUG_SETTINGS = AITDebugSettings()


CMAKELISTS_TXT_TEMPLATE = """
project({{CMAKE_PROJECT}})

# idk which version is actually needed
cmake_minimum_required(VERSION 3.20)

set(SOURCE_FILES
{{CMAKE_SOURCE_FILES}}
)

set(HEADER_FILES
{{CMAKE_HEADER_FILES}}
)

set(STANDALONE_SOURCE_FILES
{{CMAKE_STANDALONE_SOURCE_FILES}}
)

set(THIRD_PARTY_SOURCE_FILES
{{CMAKE_THIRD_PARTY_SOURCE_FILES}}
)

set(THIRD_PARTY_HEADER_FILES
{{CMAKE_THIRD_PARTY_HEADER_FILES}}
)

{% if is_linux %}
# linux only
add_custom_command(
    OUTPUT {{CMAKE_CONSTANTS_OBJ}}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ld -r -b binary -o ${CMAKE_BINARY_DIR}/{{CMAKE_CONSTANTS_OBJ}} {{CMAKE_CONSTANTS_BIN}}
    COMMAND objcopy --rename-section .data=.lrodata,alloc,load,readonly,data,contents ${CMAKE_BINARY_DIR}/{{CMAKE_CONSTANTS_OBJ}} ${CMAKE_BINARY_DIR}/{{CMAKE_CONSTANTS_OBJ}}
    DEPENDS {{CMAKE_CONSTANTS_BIN}}
)
{% endif %}

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES {{CUDA_ARCH}})

find_package(CUDAToolkit REQUIRED)

{% if cuda_static %}
set(CUDA_RUNTIME_LIBRARY Static)
{% endif %}

# this is needed to be able to pass \\ into command lline options
set(WorkaroundCmakeCompileOptions {{CMAKE_COMPILE_OPTIONS}})

# compile a supplemental library
add_library(objlib OBJECT ${SOURCE_FILES} ${THIRD_PARTY_SOURCE_FILES} {% if is_linux %}{{CMAKE_CONSTANTS_OBJ}}{% endif %})
target_include_directories(objlib PRIVATE ${HEADER_FILES} ${THIRD_PARTY_HEADER_FILES})
target_compile_options(objlib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WorkaroundCmakeCompileOptions}>)
set_target_properties(objlib PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 17)


# compile model library
add_library(model SHARED $<TARGET_OBJECTS:objlib> {% if is_linux %}{{CMAKE_CONSTANTS_OBJ}}{% endif %})
target_include_directories(model PRIVATE ${HEADER_FILES} ${THIRD_PARTY_HEADER_FILES})
target_link_libraries(model
    {% if not cuda_static %}CUDA::cudart{% endif %}
)
target_compile_options(model PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WorkaroundCmakeCompileOptions}>)
set_target_properties(model PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 17)


{% if build_standalone %}
# compile a standalone executable
add_executable(standalone $<TARGET_OBJECTS:objlib> {% if is_linux %}{{CMAKE_CONSTANTS_OBJ}}{% endif %})
target_sources(standalone PRIVATE ${STANDALONE_SOURCE_FILES})
target_include_directories(standalone PRIVATE ${HEADER_FILES} ${THIRD_PARTY_HEADER_FILES})
target_link_libraries(standalone
    {% if not cuda_static %}CUDA::cudart{% endif %}
)
target_compile_options(standalone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WorkaroundCmakeCompileOptions}>)
set_target_properties(standalone PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 17)
{% endif %}
"""


CMAKELISTS_TXT_PROFILER_TEMPLATE = """
project({{CMAKE_PROJECT}})

cmake_minimum_required(VERSION 3.20)

set(SOURCE_FILES
{{CMAKE_SOURCE_FILES}}
)

set(HEADER_FILES
{{CMAKE_HEADER_FILES}}
)

set(THIRD_PARTY_SOURCE_FILES
{{CMAKE_THIRD_PARTY_SOURCE_FILES}}
)

set(THIRD_PARTY_HEADER_FILES
{{CMAKE_THIRD_PARTY_HEADER_FILES}}
)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES {{CUDA_ARCH}})

find_package(CUDAToolkit REQUIRED)

{% if cuda_static %}
set(CUDA_RUNTIME_LIBRARY Static)
{% endif %}

# this is needed to be able to pass \\ into command lline options
set(WorkaroundCmakeCompileOptions {{CMAKE_COMPILE_OPTIONS}})

# compile a binary
add_executable(profiler ${SOURCE_FILES} ${THIRD_PARTY_SOURCE_FILES})
target_include_directories(profiler PRIVATE ${HEADER_FILES} ${THIRD_PARTY_HEADER_FILES})
target_compile_options(profiler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WorkaroundCmakeCompileOptions}>)
set_target_properties(profiler PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 17)
"""


def _run_cmd(command_line: str, timeout, custom_env: Optional[Dict[str, str]] = None):
    _LOGGER.info(f"Executing {command_line}")
    if custom_env is not None:
        for key, value in custom_env.items():
            _LOGGER.info(f"Extra environment var {key}={value}")
        environ = {**os.environ, **custom_env}
    else:
        environ = os.environ.copy()
    proc = subprocess.Popen(  # noqa: P204
        command_line,
        shell=True,
        env=environ,
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
            _LOGGER.info(f"command stdout:\n\n{stdout}")
            _LOGGER.info(f"command stderr:\n\n{stderr}")

            raise RuntimeError("command has failed.")
        else:
            _LOGGER.debug(f"command stdout:\n\n{stdout}")
            _LOGGER.debug(f"command stderr:\n\n{stderr}")


def _render_path(path: Union[Path, str]) -> str:
    # shlex.quote is designed for unit
    p = Path(path).as_posix()
    return '"' + str(p) + '"'


def _files_as_str(filenames: Union[Path, str, List[Union[Path, str]]]) -> str:
    if isinstance(filenames, str) or isinstance(filenames, Path):
        return _render_path(filenames)
    elif isinstance(filenames, list):
        return "\n".join([f"\t{_render_path(filename)}" for filename in filenames])
    else:
        raise TypeError()


class BuilderCMake:
    """BuilderCMake is a module to compile generated source code
    files into binary objects via CMake.
    """

    def __init__(self, n_cpus: int = -1, timeout: int = 180) -> None:
        self._timeout = timeout
        self._n_cpus = n_cpus

    def _build_compile_options(self) -> List[str]:
        # I don't want to move this functionality to target_def.py,
        # because target_def.py is about GNU and GNU only.

        device_compiler_options = Target.current().get_device_compiler_options()
        if is_windows():
            host_compiler_options = ["-Xcompiler=/Zc:__cplusplus"]
        else:
            host_compiler_options = [
                f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
                for opt in Target.current().get_host_compiler_options()
            ]

        compile_options = device_compiler_options + host_compiler_options

        # this is a workaround around how cmake handles \ character
        compile_options = [option.replace("\\,", "\\\\,") for option in compile_options]

        # done
        return compile_options

    def make_profilers(self, generated_profilers, workdir: Path):
        file_pairs = [f for gp in generated_profilers for f in gp]
        if not file_pairs:
            return

        # todo: combine multiple profiler in a single CMake project?
        cmake_template = jinja2.Template(CMAKELISTS_TXT_PROFILER_TEMPLATE)

        include_directories = Target.current().get_include_directories()

        compile_options = self._build_compile_options()

        # go ahead
        for source, profiler_binary in file_pairs:
            test_name = short_str(str(source))

            build_dir = Path(source).parent / test_name
            build_dir.mkdir(exist_ok=True)

            rendered = cmake_template.render(
                CMAKE_PROJECT=test_name,
                CMAKE_SOURCE_FILES=_files_as_str("../" + str(Path(source).name)),
                # # todo: this can be done once we're able to track header files
                # # properly
                # CMAKE_HEADER_FILES=_files_as_str(
                #     [Path(header).name for header in generated_sources.headers]
                # ),
                CMAKE_HEADER_FILES=_files_as_str([]),
                CMAKE_THIRD_PARTY_HEADER_FILES=_files_as_str(include_directories),
                CMAKE_THIRD_PARTY_SOURCE_FILES=_files_as_str([]),
                CMAKE_COMPILE_OPTIONS=" ".join(compile_options),
                CUDA_ARCH=Target.current()._arch,
                cuda_static=is_windows(),
                is_linux=is_linux(),
            )

            cmake_filename = build_dir / "CMakeLists.txt"
            with cmake_filename.open("w") as f:
                f.write(rendered)

            # execute cmake
            cmake_build_dir = build_dir / "build"
            cmake_cmd = Target.current().cmake()
            cmake_command_line = f"{_render_path(cmake_cmd)} -B {_render_path(cmake_build_dir)} -S {_render_path(build_dir)}"
            _run_cmd(cmake_command_line, self._timeout)

            # execute build system
            if is_windows():
                # use msbuild
                msbuild_sln_filename = cmake_build_dir / f"{test_name}.sln"
                msbuild_command_line = f"msbuild {_render_path(msbuild_sln_filename)}"
                if self._n_cpus < 0:
                    msbuild_command_line += " -m"
                else:
                    msbuild_command_line += f" -m:{self._n_cpus}"

                if Target.current()._ndebug == 1:
                    msbuild_command_line += " /property:Configuration=Release"
                else:
                    msbuild_command_line += " /property:Configuration=Debug"

                _run_cmd(msbuild_command_line, self._timeout)

                target_profiler_filename = profiler_binary
                if Target.current()._ndebug == 1:
                    compiled_profiler_filename = (
                        cmake_build_dir / "Release" / "profiler.exe"
                    )
                    shutil.copy(compiled_profiler_filename, target_profiler_filename)
                else:
                    compiled_profiler_filename = (
                        cmake_build_dir / "Debug" / "profiler.exe"
                    )
                    shutil.copy(compiled_profiler_filename, target_profiler_filename)
            else:
                # use make
                make_cmd = Target.current().make()
                make_command_line = f"{make_cmd} -C {_render_path(cmake_build_dir)}"
                if self._n_cpus < 0:
                    make_command_line += " -j"
                else:
                    make_command_line += f" -j{self._n_cpus}"

                _run_cmd(make_command_line, self._timeout)

                target_profiler_filename = profiler_binary
                compiled_profiler_filename = cmake_build_dir / "profiler"
                shutil.copy(compiled_profiler_filename, target_profiler_filename)

    def make(
        self,
        file_pairs,
        dll_name: str,
        workdir: Path,
        test_name: str,
        debug_settings: AITDebugSettings = _DEBUG_SETTINGS,
        allow_cache=False,
    ):
        # Generates a CMakeLists.txt files and builds a model and a standalone project

        if allow_cache:
            _LOGGER.warning("Caching is not yet supported")

        build_dir = Path(workdir) / test_name

        cmake_template = jinja2.Template(CMAKELISTS_TXT_TEMPLATE)

        include_directories = Target.current().get_include_directories()

        compile_options = self._build_compile_options()

        # check constants.bin
        cmake_third_party_source_files = []
        cmake_third_party_header_files = []
        constants_bin_file = build_dir / "constants.bin"

        if constants_bin_file.exists():
            if is_windows():
                resource_file = build_dir / "constants.rc"
                with resource_file.open("w") as f:
                    f.write('constant_bin CUSTOMDATA "constants.bin"')
                cmake_third_party_source_files.append("constants.rc")

                cmake_third_party_header_files.append("windll.h")
                cmake_third_party_source_files.append("windll.cu")

        # windows uses static CUDA build, linux uses dynamic one
        rendered = cmake_template.render(
            CMAKE_PROJECT=test_name,
            CMAKE_SOURCE_FILES=_files_as_str(
                [
                    Path(source).name
                    for (source, _) in file_pairs
                    if Path(source).name not in ["standalone.cu", "windll.cu"]
                ]
            ),
            # # todo: this can be done once we're able to track header files
            # # properly
            # CMAKE_HEADER_FILES=_files_as_str(
            #     [Path(header).name for header in final_sources.headers]
            # ),
            CMAKE_HEADER_FILES=_files_as_str([]),
            CMAKE_STANDALONE_SOURCE_FILES=_render_path("standalone.cu"),
            CMAKE_THIRD_PARTY_SOURCE_FILES=_files_as_str(
                cmake_third_party_source_files
            ),
            CMAKE_THIRD_PARTY_HEADER_FILES=_files_as_str(
                include_directories + cmake_third_party_header_files
            ),
            CMAKE_CONSTANTS_BIN=_render_path("constants.bin"),
            CMAKE_CONSTANTS_OBJ=_render_path("constants.obj"),
            CMAKE_COMPILE_OPTIONS=" ".join(compile_options),
            CUDA_ARCH=Target.current()._arch,
            cuda_static=is_windows(),
            is_linux=is_linux(),
            build_standalone=debug_settings.gen_standalone,
        )

        cmake_filename = build_dir / "CMakeLists.txt"
        with cmake_filename.open("w") as f:
            f.write(rendered)

        # execute cmake
        cmake_build_dir = build_dir / "build"
        cmake_cmd = Target.current().cmake()
        cmake_command_line = f"{_render_path(cmake_cmd)} -B {_render_path(cmake_build_dir)} -S {_render_path(build_dir)}"
        _run_cmd(cmake_command_line, self._timeout)

        # execute build system
        if is_windows():
            # use msbuild
            msbuild_sln_filename = cmake_build_dir / f"{test_name}.sln"
            msbuild_command_line = f"msbuild {_render_path(msbuild_sln_filename)}"
            if self._n_cpus < 0:
                msbuild_command_line += " -m"
            else:
                msbuild_command_line += f" -m:{self._n_cpus}"

            if Target.current()._ndebug == 1:
                msbuild_command_line += " /property:Configuration=Release"
            else:
                msbuild_command_line += " /property:Configuration=Debug"

            _run_cmd(msbuild_command_line, self._timeout)

            # copy
            target_library_filename = build_dir / dll_name
            target_standalone_filename = build_dir / f"{Path(dll_name).stem}.exe"
            if Target.current()._ndebug == 1:
                # copy library to where it is supposed to be
                compiled_library_filename = cmake_build_dir / "Release" / "model.dll"
                shutil.copy(compiled_library_filename, target_library_filename)

                if debug_settings.gen_standalone:
                    # copy standalone file to where it is supposed to be
                    compiled_standlone_filename = (
                        cmake_build_dir / "Release" / "standalone.exe"
                    )
                    shutil.copy(compiled_standlone_filename, target_standalone_filename)
            else:
                # copy library to where it is supposed to be
                compiled_library_filename = cmake_build_dir / "Debug" / "model.dll"
                shutil.copy(compiled_library_filename, target_library_filename)

                if debug_settings.gen_standalone:
                    # copy standalone file to where it is supposed to be
                    compiled_standlone_filename = (
                        cmake_build_dir / "Debug" / "standalone.exe"
                    )
                    shutil.copy(compiled_standlone_filename, target_standalone_filename)
        else:
            # use make
            make_cmd = Target.current().make()
            make_command_line = f"{make_cmd} -C {_render_path(cmake_build_dir)}"
            if self._n_cpus < 0:
                make_command_line += " -j"
            else:
                make_command_line += f" -j{self._n_cpus}"

            _run_cmd(make_command_line, self._timeout)

            # copy library to where it is supposed to be
            target_library_filename = build_dir / dll_name
            compiled_library_filename = cmake_build_dir / "libmodel.so"
            shutil.copy(compiled_library_filename, target_library_filename)

            if debug_settings.gen_standalone:
                # copy standalone file to where it is supposed to be
                target_standalone_filename = build_dir / (Path(dll_name).stem)
                compiled_standalone_filename = cmake_build_dir / "standalone"
                shutil.copy(compiled_standalone_filename, target_standalone_filename)
