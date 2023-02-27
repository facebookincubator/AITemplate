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

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            ) from exc

        try:
            import torch.utils

            cmake_prefix_path = torch.utils.cmake_prefix_path
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot import torch.utils. Check torch installation."
            ) from exc

        build_directory = os.path.abspath(self.build_temp)
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + build_directory,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_PREFIX_PATH=" + cmake_prefix_path,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ["--", "-j2"]

        self.build_args = build_args

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.check_call(
            ["cmake", cmake_list_dir] + cmake_args, cwd=self.build_temp, env=env
        )

        print("-" * 10, "Building extensions", "-" * 40)
        cmake_cmd = ["cmake", "--build", "."] + self.build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)
        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        lib_name = "lib" + ext.name + ".so"
        dest_path = build_temp.parents[0] / lib_name
        source_path = build_temp / lib_name
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


ext_modules = [
    CMakeExtension("ait_model"),
]

setup(
    name="fx2ait",
    version="0.2.dev1",
    description="FX2AIT: Convert PyTorch Models to AITemplate",
    zip_safe=True,
    install_requires=["torch"],  # We will need torch>=1.13
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
)
