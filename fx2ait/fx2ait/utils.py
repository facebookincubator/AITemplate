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
from aitemplate.utils.torch_utils import torch_dtype_to_string


def dtype_to_str(dtype):
    if dtype is None:
        return "float16"
    return torch_dtype_to_string(dtype)


def make_str_ait_friendly(s: str) -> str:
    if s.isalnum():
        ret = s
    else:
        ret = "".join(c if c.isalnum() else "_" for c in s)
    if ret[0].isdigit():
        ret = "_" + ret
    return ret
