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
from aitemplate.compiler.ops.gemm_universal.bmm_rcr_permute import bmm_rcr_permute
from aitemplate.compiler.ops.gemm_universal.bmm_rrr_permute import bmm_rrr_permute
from aitemplate.compiler.ops.gemm_universal.bmm_softmax_bmm import bmm_softmax_bmm
from aitemplate.compiler.ops.gemm_universal.bmm_softmax_bmm_permute import (
    bmm_softmax_bmm_permute,
)
from aitemplate.compiler.ops.gemm_universal.bmm_xxx import (
    bmm_ccc,
    bmm_ccr,
    bmm_crc,
    bmm_crr,
    bmm_rcc,
    bmm_rcr,
    bmm_rrc,
    bmm_rrr,
)
from aitemplate.compiler.ops.gemm_universal.bmm_xxx_add import (
    bmm_ccc_add,
    bmm_ccr_add,
    bmm_crc_add,
    bmm_crr_add,
    bmm_rcc_add,
    bmm_rcr_add,
    bmm_rrc_add,
    bmm_rrr_add,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias import gemm_rcr_bias
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_add import gemm_rcr_bias_add
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_add_add import (
    gemm_rcr_bias_add_add,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_add_add_relu import (
    gemm_rcr_bias_add_add_relu,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_add_relu import (
    gemm_rcr_bias_add_relu,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_fast_gelu import (
    gemm_rcr_bias_fast_gelu,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_gelu import gemm_rcr_bias_gelu
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_hardswish import (
    gemm_rcr_bias_hardswish,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_mul import gemm_rcr_bias_mul
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_mul_add import (
    gemm_rcr_bias_mul_add,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_mul_tanh import (
    gemm_rcr_bias_mul_tanh,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_permute import (
    gemm_rcr_bias_permute,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_relu import gemm_rcr_bias_relu
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_sigmoid import (
    gemm_rcr_bias_sigmoid,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_sigmoid_mul import (
    gemm_rcr_bias_sigmoid_mul,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_sigmoid_mul_tanh import (
    gemm_rcr_bias_sigmoid_mul_tanh,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_swish import (
    gemm_rcr_bias_swish,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias_tanh import gemm_rcr_bias_tanh
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_fast_gelu import gemm_rcr_fast_gelu
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_permute import gemm_rcr_permute
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_permute_elup1 import (
    gemm_rcr_permute_elup1,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rrr import gemm_rrr
from aitemplate.compiler.ops.gemm_universal.gemm_rrr_bias import gemm_rrr_bias
from aitemplate.compiler.ops.gemm_universal.gemm_rrr_bias_permute import (
    gemm_rrr_bias_permute,
)
from aitemplate.compiler.ops.gemm_universal.gemm_rrr_permute import gemm_rrr_permute
from aitemplate.compiler.ops.gemm_universal.group_gemm_rcr import group_gemm_rcr
from aitemplate.compiler.ops.gemm_universal.group_gemm_rcr_bias import (
    group_gemm_rcr_bias,
)
from aitemplate.compiler.ops.gemm_universal.group_gemm_rcr_bias_relu import (
    group_gemm_rcr_bias_relu,
)
from aitemplate.compiler.ops.gemm_universal.group_gemm_rcr_bias_sigmoid import (
    group_gemm_rcr_bias_sigmoid,
)
from aitemplate.compiler.ops.gemm_universal.perm021fc_ccr import perm021fc_ccr
from aitemplate.compiler.ops.gemm_universal.perm021fc_ccr_bias import perm021fc_ccr_bias
from aitemplate.compiler.ops.gemm_universal.perm021fc_ccr_bias_permute import (
    perm021fc_ccr_bias_permute,
)
from aitemplate.compiler.ops.gemm_universal.perm021fc_crc import perm021fc_crc
from aitemplate.compiler.ops.gemm_universal.perm021fc_crc_bias import perm021fc_crc_bias
from aitemplate.compiler.ops.gemm_universal.perm102_bmm_rcr import perm102_bmm_rcr
from aitemplate.compiler.ops.gemm_universal.perm102_bmm_rcr_bias import (
    perm102_bmm_rcr_bias,
)
from aitemplate.compiler.ops.gemm_universal.perm102_bmm_rrr import perm102_bmm_rrr
from aitemplate.compiler.ops.gemm_universal.perm102_bmm_rrr_bias import (
    perm102_bmm_rrr_bias,
)
