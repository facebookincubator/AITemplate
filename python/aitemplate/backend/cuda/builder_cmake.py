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

# a custom compile engine for CMake for CUDA backend
# handles both Windows and Linux use cases

from __future__ import annotations

import logging

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Union

import jinja2

from aitemplate.backend.compile_engine import CompileEngine
from aitemplate.backend.sources import GeneratedSourceFiles
from aitemplate.backend.target import Target

from aitemplate.utils.debug_settings import AITDebugSettings

from aitemplate.utils.misc import is_linux, is_windows


# pylint: disable=W0221,C0103


_LOGGER = logging.getLogger(__name__)
_DEBUG_SETTINGS = AITDebugSettings()


CMAKELISTS_TXT_TEMPLATE = """
project({{CMAKE_PROJECT}})

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
set(CMAKE_CUDA_ARCHITECTURES 80)

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


{% if True %}
# disable for now
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
set(CMAKE_CUDA_ARCHITECTURES 80)

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


def _run_cmd(command_line: str, timeout, custom_env: Dict[str, str] = {}):
    _LOGGER.info(f"Executing {command_line}")
    for key, value in custom_env.items():
        _LOGGER.info(f"Extra environment var {key}={value}")
    environ = {**os.environ, **custom_env}
    proc = subprocess.Popen(
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


class Builder(CompileEngine):
    """BuilderCMake is a module to compile generated source code
    files into binary objects via CMake.
    """

    def __init__(self, target: Target, n_cpus: int = -1, timeout: int = 180) -> None:
        self._timeout = timeout
        self._n_cpus = n_cpus
        self._target = target

    def make_profilers(self, generated_profilers, workdir: Path):
        file_pairs = [f for gp in generated_profilers for f in gp]
        if not file_pairs:
            return

        # todo: combine multiple profiler in a single CMake project?
        cmake_template = jinja2.Template(CMAKELISTS_TXT_PROFILER_TEMPLATE)

        include_directories = self._target.get_include_directories()
        compile_options = self._target.get_raw_compile_options()

        # a dirty hack there
        compile_options = [option.replace("\\,", "\\\\,") for option in compile_options]

        # go ahead
        for source, profiler_binary in file_pairs:
            generated_sources = GeneratedSourceFiles()
            generated_sources.add(source)

            from aitemplate.utils.misc import short_str

            test_name = short_str(str(source))

            build_dir = Path(source).parent / test_name
            build_dir.mkdir(exist_ok=True)

            # render
            def _files_as_str(filenames) -> str:
                return "\n".join(
                    [f"\t{_render_path(filename)}" for filename in filenames]
                )

            rendered = cmake_template.render(
                CMAKE_PROJECT=test_name,
                CMAKE_SOURCE_FILES=_files_as_str(
                    [
                        "../" + str(Path(source).name)
                        for source in generated_sources.sources
                    ]
                ),
                CMAKE_HEADER_FILES=_files_as_str(
                    [Path(header).name for header in generated_sources.headers]
                ),
                CMAKE_THIRD_PARTY_HEADER_FILES=_files_as_str(include_directories),
                CMAKE_THIRD_PARTY_SOURCE_FILES=_files_as_str([]),
                CMAKE_COMPILE_OPTIONS=" ".join(compile_options),
                cuda_static=is_windows(),
                is_linux=is_linux(),
            )

            cmake_filename = build_dir / "CMakeLists.txt"
            with cmake_filename.open("w") as f:
                f.write(rendered)

            # execute cmake
            cmake_build_dir = build_dir / "build"
            cmake_command_line = f"{_render_path(self._target.cmake())} -B {_render_path(cmake_build_dir)} -S {_render_path(build_dir)}"
            _run_cmd(cmake_command_line, self._timeout)

            # execute build system
            if is_windows():
                msbuild_sln_filename = cmake_build_dir / f"{test_name}.sln"
                msbuild_command_line = f"msbuild {_render_path(msbuild_sln_filename)}"
                if self._n_cpus < 0:
                    msbuild_command_line += " -m"
                else:
                    msbuild_command_line += f" -m:{self._n_cpus}"

                if self._target._ndebug == 1:
                    msbuild_command_line += " /property:Configuration=Release"
                else:
                    msbuild_command_line += " /property:Configuration=Debug"

                _run_cmd(msbuild_command_line, self._timeout)

                target_profiler_filename = profiler_binary
                if self._target._ndebug == 1:
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
                make_command_line = (
                    f"{self._target.make()} -C {_render_path(cmake_build_dir)}"
                )
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
        generated_sources: GeneratedSourceFiles,
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

        include_directories = self._target.get_include_directories()
        compile_options = self._target.get_raw_compile_options()

        # a dirty hack there
        compile_options = [option.replace("\\,", "\\\\,") for option in compile_options]

        #
        final_sources = GeneratedSourceFiles()
        final_sources.add(generated_sources)

        # check constants.bin
        cmake_third_party_source_files = []
        cmake_third_party_header_files = []
        constants_bin_file = build_dir / "constants.bin"

        if constants_bin_file.exists():
            # todo: replace with is_msvc()
            if is_windows():
                resource_file = build_dir / "constants.rc"
                with resource_file.open("w") as f:
                    f.write('constant_bin CUSTOMDATA "constants.bin"')
                cmake_third_party_source_files.append("constants.rc")

                cmake_third_party_header_files.append("windll.h")
                cmake_third_party_source_files.append("windll.cu")
            else:
                # CMake uses ld for linking constants.bin
                pass

        # # it is also possible to render a special .cpp file,
        # #   something like this.
        # from aitemplate.utils.cmake_utils import constants_bin_2_cpp
        # constants_cpp_file = build_dir / "constants.cpp"
        # constants_bin_2_cpp(constants_bin_file, constants_cpp_file)
        # # and then add it to our sources
        # final_sources.add(str(constants_cpp_file))

        # windows uses static CUDA build, linux uses dynamic one

        def _files_as_str(filenames) -> str:
            return "\n".join([f"\t{_render_path(filename)}" for filename in filenames])

        rendered = cmake_template.render(
            CMAKE_PROJECT=test_name,
            CMAKE_SOURCE_FILES=_files_as_str(
                [
                    Path(source).name
                    for source in final_sources.sources
                    if Path(source).name not in ["standalone.cu", "windll.cu"]
                ]
            ),
            CMAKE_HEADER_FILES=_files_as_str(
                [Path(header).name for header in final_sources.headers]
            ),
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
            cuda_static=is_windows(),
            is_linux=is_linux(),
        )

        cmake_filename = build_dir / "CMakeLists.txt"
        with cmake_filename.open("w") as f:
            f.write(rendered)

        # execute cmake
        cmake_build_dir = build_dir / "build"
        cmake_command_line = f"{_render_path(self._target.cmake())} -B {_render_path(cmake_build_dir)} -S {_render_path(build_dir)}"
        _run_cmd(cmake_command_line, self._timeout)

        # execute build system
        if is_windows():
            msbuild_sln_filename = cmake_build_dir / f"{test_name}.sln"
            msbuild_command_line = f"msbuild {_render_path(msbuild_sln_filename)}"
            if self._n_cpus < 0:
                msbuild_command_line += " -m"
            else:
                msbuild_command_line += f" -m:{self._n_cpus}"

            if self._target._ndebug == 1:
                msbuild_command_line += " /property:Configuration=Release"
            else:
                msbuild_command_line += " /property:Configuration=Debug"

            _run_cmd(msbuild_command_line, self._timeout)

            target_library_filename = build_dir / dll_name
            if self._target._ndebug == 1:
                compiled_library_filename = cmake_build_dir / "Release" / "model.dll"
                shutil.copy(compiled_library_filename, target_library_filename)
            else:
                compiled_library_filename = cmake_build_dir / "Debug" / "model.dll"
                shutil.copy(compiled_library_filename, target_library_filename)
        else:
            # use make
            make_command_line = (
                f"{self._target.make()} -C {_render_path(cmake_build_dir)}"
            )
            if self._n_cpus < 0:
                make_command_line += " -j"
            else:
                make_command_line += f" -j{self._n_cpus}"

            _run_cmd(make_command_line, self._timeout)

            target_library_filename = build_dir / dll_name
            compiled_library_filename = cmake_build_dir / "libmodel.so"
            shutil.copy(compiled_library_filename, target_library_filename)
