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
from .container import ModuleDict, ModuleList, Sequential
from .embedding import BertEmbeddings, Embedding
from .module import Module
from .conv2d import *
from .conv3d import *
from .linear import *
from .padding import *
from .pool2d import *
from .fpn_proposal import FPNProposal
from .proposal import Proposal
from .roi_ops import *
from .upsample import *
from .view_ops import *
from .attention import CrossAttention, FlashAttention, MultiheadAttention
from .identity import Identity
from .multiscale_attention import MultiScaleBlock
from .vanilla_attention import (
    vanilla_attention,
    VanillaCrossAttention,
    VanillaMultiheadAttention,
)
from .dropout import *
from .layer_norm import *
from .group_norm import *
from .dual_gemm import T5DenseGatedGeluDense
