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
Functions for working with torch Tensors.
AITemplate doesn't depend on PyTorch, but it exposes
many APIs that work with torch Tensors anyways.

The functions in this file may assume that
`import torch` will work.
"""


def torch_dtype_to_string(dtype):
    import torch

    dtype_to_str = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    if dtype not in dtype_to_str:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(dtype_to_str.keys())}"
        )
    return dtype_to_str[dtype]
