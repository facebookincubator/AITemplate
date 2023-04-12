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
from aitemplate.frontend.nn.container import ModuleDict, ModuleList, Sequential
from aitemplate.frontend.nn.embedding import BertEmbeddings, Embedding
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.conv1d import *
from aitemplate.frontend.nn.conv2d import *
from aitemplate.frontend.nn.conv3d import *
from aitemplate.frontend.nn.linear import *
from aitemplate.frontend.nn.padding import *
from aitemplate.frontend.nn.pool2d import *
from aitemplate.frontend.nn.fpn_proposal import FPNProposal
from aitemplate.frontend.nn.proposal import Proposal
from aitemplate.frontend.nn.roi_ops import *
from aitemplate.frontend.nn.upsample import *
from aitemplate.frontend.nn.view_ops import *
from aitemplate.frontend.nn.attention import (
    CrossAttention,
    FlashAttention,
    MultiheadAttention,
    ScaledDotProductAttention,
)
from aitemplate.frontend.nn.identity import Identity
from aitemplate.frontend.nn.multiscale_attention import MultiScaleBlock
from aitemplate.frontend.nn.vanilla_attention import (
    vanilla_attention,
    VanillaCrossAttention,
    VanillaMultiheadAttention,
)
from aitemplate.frontend.nn.dropout import *
from aitemplate.frontend.nn.layer_norm import *
from aitemplate.frontend.nn.group_norm import *
from aitemplate.frontend.nn.dual_gemm import T5DenseGatedGeluDense
