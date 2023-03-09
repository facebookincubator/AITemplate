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
ROCM tensor ops module init
"""
from aitemplate.backend.rocm.tensor import (  # noqa
    argmax,
    batch_gather,
    concatenate,
    concatenate_tanh,
    dynamic_slice,
    permute021,
    permute0213,
    permute102,
    permute210,
    slice_reshape_scatter,
    slice_scatter,
    split,
    topk,
)
