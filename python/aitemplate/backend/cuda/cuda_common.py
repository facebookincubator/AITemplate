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
CUDA common functions for codegen.
"""
from typing import Dict

DTYPE_TO_CUDATYPE: Dict[str, str] = {
    "float16": "half",
    "float32": "float",
    "float": "float",
    "int64": "int64_t",
}


DTYPE_TO_CUTLASSTYPE: Dict[str, str] = {
    "float16": "cutlass::half_t",
    "float": "float",
}


def dtype_to_cuda_type(dtype: str):
    """Returns the corresponding cuda type."""
    cuda_type = DTYPE_TO_CUDATYPE.get(dtype)

    if cuda_type is None:
        raise NotImplementedError("CUDA - Unsupported dtype: {}".format(dtype))
    return cuda_type


def dtype_to_cutlass_type(dtype: str):
    """Returns the corresponding cutlass type."""
    cutlass_type = DTYPE_TO_CUTLASSTYPE.get(dtype)

    if cutlass_type is None:
        raise NotImplementedError("CUDA - Unsupported dtype: {}".format(dtype))
    return cutlass_type
