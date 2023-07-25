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

import glob
import os

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    print("Compiling extensions with following flags:")
    debug_mode = os.getenv("DEBUG", "0") == "1"
    print(f"  DEBUG: {debug_mode}")
    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    print(f"  NVCC_FLAGS: {nvcc_flags}")
    if nvcc_flags == "":
        nvcc_flags = []
    else:
        nvcc_flags = nvcc_flags.split(" ")
    extra_compile_args = {"cxx": [], "nvcc": nvcc_flags}

    if debug_mode:
        print("Compiling in debug mode")
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [
                f for f in nvcc_flags if not ("-O" in f or "-g" in f)
            ]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "fx2ait", "csrc")

    src = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    inc = [extensions_dir]
    inc += [os.path.abspath(os.path.join(this_dir, "../static/include"))]
    inc += [os.path.abspath(os.path.join(this_dir, "../3rdparty/picojson"))]
    define_macros = []

    ext_modules = [
        CUDAExtension(
            name="fx2ait.libait_model",
            sources=src,
            include_dirs=inc,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="fx2ait",
    version="0.2.dev1",
    description="FX2AIT: Convert PyTorch Models to AITemplate",
    zip_safe=False,
    install_requires=["torch"],  # We will need torch>=1.13
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
