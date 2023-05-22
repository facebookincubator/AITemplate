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
    ControlNetModel as ait_ControlNetModel,
)
from .util import mark_output


def map_controlnet_params(pt_mod, dim):
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
    params_ait["controlnet_cond_embedding_conv_in_weight"] = torch.nn.functional.pad(
        params_ait["controlnet_cond_embedding_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
    )
    params_ait["arange"] = (
        torch.arange(start=0, end=320 // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


# was used to debug controlnet_cond_embedding
# def compile_controlnet_conditioning_embedding():
#     controlnet_cond_embedding = ait_ControlNetConditioningEmbedding(256)
#     controlnet_cond_embedding.name_parameter_tensor()
#     controlnet_condition_ait = Tensor(
#         [2, 512, 512, 3], name="input0", is_input=True
#     )
#     Y = controlnet_cond_embedding(controlnet_condition_ait)
#     mark_output(Y)
#     target = detect_target(
#         use_fp16_acc=True, convert_conv_to_gemm=True
#     )
#     compile_model(Y, target, "./tmp", "ControlNetConditioningEmbedding", constants=None)


def compile_controlnet(
    pt_mod,
    batch_size=2,
    height=512,  # (512,1024),
    width=512,  # (512,1024),
    clip_chunks=1,
    dim=320,
    hidden_dim=768,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    model_name="ControlNetModel",
    constants=False,
):
    batch_size = batch_size * 2  # double batch size for unet
    ait_mod = ait_ControlNetModel()
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_controlnet_params(pt_mod, dim)
    # batch_size = (batch_size[0], batch_size[1] * 2) #double batch size for unet
    clip_batch_size = IntVar(values=(1, 8), name="batch_size")
    # height_d = IntVar(values=list((height[0]//8, height[1]//8)), name="height_d")
    # width_d = IntVar(values=list((width[0]//8, width[1]//8)), name="width_d")
    # height_c = IntVar(values=list((height[0], height[1])), name="height_c")
    # width_c = IntVar(values=list((width[0], width[1])), name="width_c")
    clip_chunks = 77, 77 * clip_chunks
    embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height // 8, width // 8, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [clip_batch_size, embedding_size, hidden_dim], name="input2", is_input=True
    )
    controlnet_condition_ait = Tensor(
        [batch_size, height, width, 3], name="input3", is_input=True
    )

    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        text_embeddings_pt_ait,
        controlnet_condition_ait,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, "./tmp", model_name, constants=params_ait if constants else None
    )
