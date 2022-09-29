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
from . import lib_template, target_def, utils
from .common import *
from .conv2d import *
from .gemm import *
from .pool2d import *
from .view_ops import *
from .elementwise import *
from .tensor import *
from .normalization import softmax
from .upsample import *
from .vision_ops import *
from .normalization import groupnorm, groupnorm_swish, layernorm
