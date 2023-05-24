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
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.controlnet_unet_2d_condition import (
    ControlNetUNet2DConditionModel as ait_ControlNetUNet2DConditionModel,
)
from ..modeling.unet_2d_condition import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import mark_output


def map_unet_params(pt_mod, dim):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def compile_unet(
    pt_mod,
    batch_size=(1, 8),
    height=(64, 2048),
    width=(64, 2048),
    clip_chunks=1,
    dim=320,
    hidden_dim=1024,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    controlnet=True,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
    model_name="UNet2DConditionModel",
    use_linear_projection=False,
    constants=True,
):
    if controlnet:
        ait_mod = ait_ControlNetUNet2DConditionModel(
            sample_size=64,
            cross_attention_dim=hidden_dim,
            attention_head_dim=attention_head_dim,
            use_linear_projection=use_linear_projection,
        )
    else:
        ait_mod = ait_UNet2DConditionModel(
            sample_size=64,
            cross_attention_dim=hidden_dim,
            attention_head_dim=attention_head_dim,
            use_linear_projection=use_linear_projection,
        )
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pt_mod.eval()
    params_ait = map_unet_params(pt_mod, dim)
    if controlnet:
        # static sizes only for now
        batch_size = batch_size[0] * 2  # double batch size for unet
        height = height[0] // 8
        width = width[0] // 8
        height_d = height
        width_d = width
    else:
        batch_size = (batch_size[0], batch_size[1] * 2)  # double batch size for unet
        batch_size = IntVar(values=list(batch_size), name="batch_size")
        height = height[0] // 8, height[1] // 8
        width = width[0] // 8, width[1] // 8
        height_d = IntVar(values=list(height), name="height")
        width_d = IntVar(values=list(width), name="width")
    clip_chunks = 77, 77 * clip_chunks
    embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim], name="input2", is_input=True
    )
    if controlnet:
        down_block_residual_0 = Tensor(
            [batch_size, height, width, 320],
            name="down_block_residual_0",
            is_input=True,
        )
        down_block_residual_1 = Tensor(
            [batch_size, height, width, 320],
            name="down_block_residual_1",
            is_input=True,
        )
        down_block_residual_2 = Tensor(
            [batch_size, height, width, 320],
            name="down_block_residual_2",
            is_input=True,
        )
        down_block_residual_3 = Tensor(
            [batch_size, height // 2, width // 2, 320],
            name="down_block_residual_3",
            is_input=True,
        )
        down_block_residual_4 = Tensor(
            [batch_size, height // 2, width // 2, 640],
            name="down_block_residual_4",
            is_input=True,
        )
        down_block_residual_5 = Tensor(
            [batch_size, height // 2, width // 2, 640],
            name="down_block_residual_5",
            is_input=True,
        )
        down_block_residual_6 = Tensor(
            [batch_size, height // 4, width // 4, 640],
            name="down_block_residual_6",
            is_input=True,
        )
        down_block_residual_7 = Tensor(
            [batch_size, height // 4, width // 4, 1280],
            name="down_block_residual_7",
            is_input=True,
        )
        down_block_residual_8 = Tensor(
            [batch_size, height // 4, width // 4, 1280],
            name="down_block_residual_8",
            is_input=True,
        )
        down_block_residual_9 = Tensor(
            [batch_size, height // 8, width // 8, 1280],
            name="down_block_residual_9",
            is_input=True,
        )
        down_block_residual_10 = Tensor(
            [batch_size, height // 8, width // 8, 1280],
            name="down_block_residual_10",
            is_input=True,
        )
        down_block_residual_11 = Tensor(
            [batch_size, height // 8, width // 8, 1280],
            name="down_block_residual_11",
            is_input=True,
        )
        mid_block_residual = Tensor(
            [batch_size, height // 8, width // 8, 1280],
            name="mid_block_residual",
            is_input=True,
        )
    else:
        mid_block_additional_residual = None
        down_block_additional_residuals = None

    if controlnet:
        Y = ait_mod(
            latent_model_input_ait,
            timesteps_ait,
            text_embeddings_pt_ait,
            down_block_residual_0,
            down_block_residual_1,
            down_block_residual_2,
            down_block_residual_3,
            down_block_residual_4,
            down_block_residual_5,
            down_block_residual_6,
            down_block_residual_7,
            down_block_residual_8,
            down_block_residual_9,
            down_block_residual_10,
            down_block_residual_11,
            mid_block_residual,
        )
    else:
        Y = ait_mod(
            latent_model_input_ait,
            timesteps_ait,
            text_embeddings_pt_ait,
            mid_block_additional_residual,
            down_block_additional_residuals,
        )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, "./tmp", model_name, constants=params_ait if constants else None
    )
