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
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, exec-used

import os
import shutil

from setuptools import find_packages, setup
from setuptools.dist import Distribution

# flake8: noqa

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "aitemplate", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


# temp copy 3rdparty libs to build dir
shutil.copytree("../3rdparty", "./aitemplate/3rdparty")
shutil.copytree("../static", "./aitemplate/static")
shutil.copytree("../licenses", "./aitemplate/licenses")


def gen_file_list(srcs, f_cond):
    file_list = []
    for src in srcs:
        for root, _, files in os.walk(src):
            value = []
            for file in files:
                if f_cond(file):
                    path = os.path.join(root, file)
                    value.append(path.replace("aitemplate/", ""))
            file_list.extend(value)
    return file_list


def gen_cutlass_list():
    srcs = [
        "aitemplate/3rdparty/cutlass/include",
        "aitemplate/3rdparty/cutlass/examples",
        "aitemplate/3rdparty/cutlass/tools/util/include",
    ]
    f_cond = (
        lambda x: True
        if x.endswith(".h") or x.endswith(".cuh") or x.endswith(".hpp")
        else False
    )
    return gen_file_list(srcs, f_cond)


def gen_cutlass_lib_list():
    srcs = ["aitemplate/3rdparty/cutlass/tools/library/scripts"]
    f_cond = lambda x: True
    return gen_file_list(srcs, f_cond)


def gen_cub_list():
    srcs = ["aitemplate/3rdparty/cub/cub"]
    f_cond = lambda x: True if x.endswith(".h") or x.endswith(".cuh") else False
    return gen_file_list(srcs, f_cond)


def gen_ck_list():
    srcs = [
        "aitemplate/3rdparty/composable_kernel/include",
        "aitemplate/3rdparty/composable_kernel/library/include/ck/library/utility",
    ]
    f_cond = lambda x: True if x.endswith(".h") or x.endswith(".hpp") else False
    return gen_file_list(srcs, f_cond)


def gen_flash_attention_list():
    srcs = [
        "aitemplate/backend/cuda/attention/src",
        "aitemplate/backend/cuda/attention/src/fmha",
    ]
    f_cond = lambda x: True if x.endswith(".h") or x.endswith(".cuh") else False
    return gen_file_list(srcs, f_cond)


def gen_static_list():
    srcs = [
        "aitemplate/static",
    ]
    f_cond = lambda x: True if x.endswith(".h") or x.endswith(".cpp") else False
    return gen_file_list(srcs, f_cond)


def gen_utils_file_list():
    srcs = ["aitemplate/utils"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)


def gen_backend_common_file_list():
    srcs = ["aitemplate/backend"]
    f_cond = lambda x: True if x.endswith(".py") or x.endswith(".cuh") else False
    return gen_file_list(srcs, f_cond)


def gen_license_file_list():
    srcs = ["aitemplate/licenses"]
    f_cond = lambda x: True
    return gen_file_list(srcs, f_cond)


setup_kwargs = {}
include_libs = True
wheel_include_libs = True


setup(
    name="aitemplate",
    version=__version__,
    description="AITemplate: Make Templates Great for AI",
    zip_safe=True,
    install_requires=["jinja2", "numpy", "sympy"],
    packages=find_packages(),
    package_data={
        "aitemplate": [
            "backend/cuda/elementwise/custom_math.cuh",
            "backend/cuda/layernorm_sigmoid_mul/layernorm_sigmoid_mul_kernel.cuh",
            "backend/cuda/groupnorm/groupnorm_kernel.cuh",
            "backend/cuda/groupnorm/layer_norm.cuh",
            "backend/cuda/softmax/softmax.cuh",
            "backend/cuda/vision_ops/nms/batched_nms_kernel.cuh",
            "backend/cuda/vision_ops/nms/nms_kernel.cuh",
            "backend/cuda/vision_ops/roi_ops/multi_level_roi_align.cuh",
            "backend/rocm/elementwise/custom_math.h",
        ]
        + gen_utils_file_list()
        + gen_cutlass_list()
        + gen_cutlass_lib_list()
        + gen_cub_list()
        + gen_ck_list()
        + gen_flash_attention_list()
        + gen_static_list()
        + gen_backend_common_file_list()
        + gen_license_file_list(),
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    **setup_kwargs,
)

# remove temp
shutil.rmtree("./aitemplate/3rdparty")
shutil.rmtree("./aitemplate/static")
shutil.rmtree("./aitemplate/licenses")
