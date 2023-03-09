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
common functions for conv2d op with few channels(< 8)
"""

from aitemplate.backend.cuda.conv2d import common
from aitemplate.utils import alignment


def extract_config(func_attrs, dtype="float16"):
    def apply_special_config(func_attrs, op):
        import cutlass_lib

        x = func_attrs["inputs"][0]
        in_ch = x._attrs["shape"][-1]._attrs["values"][0]

        # Make sure to use NoneGroup here. Otherwise, we'll generate Conv2dGroupFprop,
        # which doesn't have template specializations for either of the iterator
        # algorithms below, resulting in "incomplete type is not allowed" errors.
        op.group_mode = cutlass_lib.library.GroupMode.NoneGroup
        if in_ch == 3:
            # By default we don't use it since the perf is worse than pad4+fixchannel
            op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FewChannels
            op.A.alignment = 1
            op.B.alignment = 1
            op.tile_description.stages = 2
        elif in_ch in alignment.get_alignments(dtype):
            op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FixedChannels
            op.A.alignment = in_ch
            op.B.alignment = in_ch
            op.tile_description.stages = 3

        return op

    return common.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
        skip_simt_kernels=True,
        f_apply_special_config=apply_special_config,
    )
