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
Nms family ops.
"""
from aitemplate.compiler.ops.vision_ops.nms.batched_nms import batched_nms
from aitemplate.compiler.ops.vision_ops.nms.efficient_nms import efficient_nms
from aitemplate.compiler.ops.vision_ops.nms.nms import nms


__all__ = ["batched_nms", "nms", "efficient_nms"]
