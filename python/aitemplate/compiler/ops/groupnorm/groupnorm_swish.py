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

from aitemplate.compiler.ops.groupnorm.groupnorm import group_norm


class group_norm_swish(group_norm):
    """Standalone group norm op.
    The grouped dim must be the last dim of the input tensor.
    """

    def __init__(self, num_groups: int, num_channels: int) -> None:
        super().__init__(num_groups, num_channels)
        self._attrs["op"] = "groupnorm_swish"
