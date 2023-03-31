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

KEYS = [
    "op",
    "depth",
    "nop",
    "has_profiler",
    "epilogue",
    "epilogue_alignment",
    "split_k",
    "permute_shape",
]


def op_to_content(op):
    # TODO (XXX): Add op specialized attrs here, like gemm/conv
    content = {}
    for k in KEYS:
        v = op._attrs.get(k)
        if v is not None and v != "":
            content[k] = v

    if op._attrs["op"] == "fused_elementwise":
        content["func"] = ", ".join(
            [str(x._attrs["func"]) for x in op._attrs["elementwise_ops"]]
        )
    return content
