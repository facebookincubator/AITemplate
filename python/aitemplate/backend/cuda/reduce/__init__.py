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
CUDA reduce module init
"""
from aitemplate.backend.cuda.reduce import (
    reduce_3d,
    reduce_common,
    reduce_mean,
    reduce_sum,
    var,
    vector_norm,
)

__all__ = [
    "reduce_3d",
    "reduce_common",
    "reduce_mean",
    "reduce_sum",
    "var",
    "vector_norm",
]
