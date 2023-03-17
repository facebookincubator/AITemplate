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
# flake8: noqa
"""
Rocm backend init.
"""
from aitemplate.backend.rocm import lib_template, target_def, utils
from aitemplate.backend.rocm.common import *
from aitemplate.backend.rocm.conv2d import *
from aitemplate.backend.rocm.embedding import *
from aitemplate.backend.rocm.gemm import *
from aitemplate.backend.rocm.pool2d import *
from aitemplate.backend.rocm.view_ops import *
from aitemplate.backend.rocm.elementwise import *
from aitemplate.backend.rocm.tensor import *
from aitemplate.backend.rocm.normalization import softmax
from aitemplate.backend.rocm.upsample import *
from aitemplate.backend.rocm.vision_ops import *
from aitemplate.backend.rocm.padding import *
from aitemplate.backend.rocm.normalization import groupnorm, groupnorm_swish, layernorm
