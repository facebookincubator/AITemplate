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
reduce module init
"""
from aitemplate.compiler.ops.tensor.argmax import argmax
from aitemplate.compiler.ops.tensor.batch_gather import batch_gather
from aitemplate.compiler.ops.tensor.chunk import chunk
from aitemplate.compiler.ops.tensor.concatenate import concatenate
from aitemplate.compiler.ops.tensor.concatenate_tanh import concatenate_tanh
from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice
from aitemplate.compiler.ops.tensor.expand import expand
from aitemplate.compiler.ops.tensor.full import full
from aitemplate.compiler.ops.tensor.gather import gather
from aitemplate.compiler.ops.tensor.jagged_to_padded_dense import jagged_to_padded_dense
from aitemplate.compiler.ops.tensor.masked_select import masked_select
from aitemplate.compiler.ops.tensor.padded_dense_to_jagged import padded_dense_to_jagged
from aitemplate.compiler.ops.tensor.permute import permute
from aitemplate.compiler.ops.tensor.permute021 import permute021
from aitemplate.compiler.ops.tensor.permute0213 import permute0213
from aitemplate.compiler.ops.tensor.permute102 import permute102
from aitemplate.compiler.ops.tensor.permute210 import permute210
from aitemplate.compiler.ops.tensor.size import size
from aitemplate.compiler.ops.tensor.slice_reshape_scatter import slice_reshape_scatter
from aitemplate.compiler.ops.tensor.slice_scatter import slice_scatter
from aitemplate.compiler.ops.tensor.split import split
from aitemplate.compiler.ops.tensor.topk import topk
from aitemplate.compiler.ops.tensor.transpose import transpose
