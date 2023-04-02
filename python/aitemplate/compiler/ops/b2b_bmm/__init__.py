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
B2B Bmm ops.
"""

from aitemplate.compiler.ops.b2b_bmm.classic_b2b_bmm import classic_b2b_bmm
from aitemplate.compiler.ops.b2b_bmm.fmha_style_b2b_bmm import fmha_style_b2b_bmm
from aitemplate.compiler.ops.b2b_bmm.grouped_fmha_style_b2b_bmm import (
    grouped_fmha_style_b2b_bmm,
)
