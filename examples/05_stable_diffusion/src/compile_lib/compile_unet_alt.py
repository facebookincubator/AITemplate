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
import sys

import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.embeddings import Timesteps
from ..modeling.unet_2d_condition import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import torch_dtype_from_str


def map_unet(
    pt_mod, in_channels=None, conv_in_key=None, dim=320, device="cuda", dtype="float16"
):
    if in_channels is not None and conv_in_key is None:
        raise ValueError(
            "conv_in_key must be specified if in_channels is not None for padding"
        )
    if not isinstance(pt_mod, dict):
        pt_params = dict(pt_mod.named_parameters())
    else:
        pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "")
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
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

    if conv_in_key is not None:
        if in_channels % 4 != 0:
            pad_by = 4 - (in_channels % 4)
            params_ait[conv_in_key] = torch.functional.F.pad(
                params_ait[conv_in_key], (0, pad_by)
            )

    params_ait["arange"] = torch.arange(start=0, end=dim // 2, dtype=torch.float32).to(
        device, dtype=torch_dtype_from_str(dtype)
    )
    return params_ait


def compile_timestep_embedder(
    dim=256,
    flip_sin_to_cos=True,
    downscale_freq_shift=0,
    work_dir="./tmp",
    model_name="Timesteps",
):
    timesteps = Timesteps(
        dim, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=downscale_freq_shift
    )

    timestep = Tensor([1], name="timestep", is_input=True)

    Y = timesteps(timestep)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "time_embed"
    shape = [d._attrs["values"] for d in Y._attrs["shape"]]
    print(f'AIT {Y._attrs["name"]} shape: {shape}')
    constants = {"arange": torch.arange(start=0, end=dim // 2, dtype=torch.float16)}

    target = detect_target(use_fp16_acc=True, convert_conv_to_gemm=True)
    dll_name = model_name + ".dll" if sys.platform == "win32" else model_name + ".so"
    compile_model(Y, target, work_dir, model_name, constants=constants, dll_name=dll_name)


def compile_unet(
    pt_mod,
    batch_size=(1, 8),
    height=(64, 2048),
    width=(64, 2048),
    clip_chunks=1,
    work_dir="./tmp",
    dim=320,
    hidden_dim=1024,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    controlnet=False,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
    model_name="UNet2DConditionModel",
    use_linear_projection=False,
    constants=True,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    in_channels=4,
    out_channels=4,
    sample_size=64,
    class_embed_type=None,
    num_class_embeds=None,
    only_cross_attention=[True, True, True, False],
    down_factor=8,
    time_embedding_dim=None,
    conv_in_kernel: int = 3,
    projection_class_embeddings_input_dim=None,
    addition_embed_type=None,
    transformer_layers_per_block=[1, 1, 1, 1],
    dtype="float16",
):
    xl = False
    if projection_class_embeddings_input_dim is not None:
        xl = True
    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(block_out_channels)
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(
            down_block_types
        )
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(down_block_types)

    ait_mod = ait_UNet2DConditionModel(
        sample_size=sample_size,
        cross_attention_dim=hidden_dim,
        attention_head_dim=attention_head_dim,
        use_linear_projection=use_linear_projection,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        class_embed_type=class_embed_type,
        num_class_embeds=num_class_embeds,
        only_cross_attention=only_cross_attention,
        time_embedding_dim=time_embedding_dim,
        conv_in_kernel=conv_in_kernel,
        projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
        addition_embed_type=addition_embed_type,
        transformer_layers_per_block=transformer_layers_per_block,
        dtype=dtype,
    )
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pt_mod.eval()
    params_ait = map_unet(
        pt_mod,
        dim=dim,
        in_channels=in_channels,
        conv_in_key="conv_in_weight",
        dtype=dtype,
    )

    static_shape = width[0] == width[1] and height[0] == height[1]

    if static_shape:
        height = height[0] // down_factor
        width = width[0] // down_factor
        height_d = height
        width_d = width
        height_1_d = height
        width_1_d = width
        height_2 = height // 2
        width_2 = width // 2
        height_4 = height // 4
        width_4 = width // 4
        height_8 = height // 8
        width_8 = width // 8
        height_2_d = height_2
        width_2_d = width_2
        height_4_d = height_4
        width_4_d = width_4
        height_8_d = height_8
        width_8_d = width_8
    else:
        height = [x // down_factor for x in height]
        width = [x // down_factor for x in width]
        height_d = IntVar(values=list(height), name="height_d")
        width_d = IntVar(values=list(width), name="width_d")
        height_1_d = IntVar(values=list(height), name="height_1_d")
        width_1_d = IntVar(values=list(width), name="width_1_d")
        height_2 = [x // 2 for x in height]
        width_2 = [x // 2 for x in width]
        height_4 = [x // 4 for x in height]
        width_4 = [x // 4 for x in width]
        height_8 = [x // 8 for x in height]
        width_8 = [x // 8 for x in width]
        height_2_d = IntVar(values=list(height_2), name="height_2_d")
        width_2_d = IntVar(values=list(width_2), name="width_2_d")
        height_4_d = IntVar(values=list(height_4), name="height_4_d")
        width_4_d = IntVar(values=list(width_4), name="width_4_d")
        height_8_d = IntVar(values=list(height_8), name="height_8_d")
        width_8_d = IntVar(values=list(width_8), name="width_8_d")

    batch_size = batch_size[0], batch_size[1] * 2  # double batch size for unet
    batch_size = IntVar(values=list(batch_size), name="batch_size") if detect_target().name() == "cuda" else 2

    if static_shape:
        embedding_size = 77
    else:
        clip_chunks = 77, 77 * clip_chunks
        embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, in_channels],
        name="latent_model_input",
        is_input=True,
        dtype=dtype,
    )
    timesteps_ait = Tensor([batch_size], name="timesteps", is_input=True, dtype=dtype)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim],
        name="encoder_hidden_states",
        is_input=True,
        dtype=dtype,
    )

    class_labels = None
    # TODO: better way to handle this, enables class_labels for x4-upscaler
    if in_channels == 7:
        class_labels = Tensor(
            [batch_size], name="class_labels", dtype="int64", is_input=True
        )

    add_embeds = None
    if xl:
        add_embeds = Tensor(
            [batch_size, projection_class_embeddings_input_dim],
            name="add_embeds",
            is_input=True,
            dtype=dtype,
        )

    down_block_residual_0 = None
    down_block_residual_1 = None
    down_block_residual_2 = None
    down_block_residual_3 = None
    down_block_residual_4 = None
    down_block_residual_5 = None
    down_block_residual_6 = None
    down_block_residual_7 = None
    down_block_residual_8 = None
    down_block_residual_9 = None
    down_block_residual_10 = None
    down_block_residual_11 = None
    mid_block_residual = None
    if controlnet:
        down_block_residual_0 = Tensor(
            [batch_size, height_1_d, width_1_d, block_out_channels[0]],
            name="down_block_residual_0",
            is_input=True,
        )
        down_block_residual_1 = Tensor(
            [batch_size, height_1_d, width_1_d, block_out_channels[0]],
            name="down_block_residual_1",
            is_input=True,
        )
        down_block_residual_2 = Tensor(
            [batch_size, height_1_d, width_1_d, block_out_channels[0]],
            name="down_block_residual_2",
            is_input=True,
        )
        down_block_residual_3 = Tensor(
            [batch_size, height_2_d, width_2_d, block_out_channels[0]],
            name="down_block_residual_3",
            is_input=True,
        )
        down_block_residual_4 = Tensor(
            [batch_size, height_2_d, width_2_d, block_out_channels[1]],
            name="down_block_residual_4",
            is_input=True,
        )
        down_block_residual_5 = Tensor(
            [batch_size, height_2_d, width_2_d, block_out_channels[1]],
            name="down_block_residual_5",
            is_input=True,
        )
        down_block_residual_6 = Tensor(
            [batch_size, height_4_d, width_4_d, block_out_channels[1]],
            name="down_block_residual_6",
            is_input=True,
        )
        down_block_residual_7 = Tensor(
            [batch_size, height_4_d, width_4_d, block_out_channels[2]],
            name="down_block_residual_7",
            is_input=True,
        )
        down_block_residual_8 = Tensor(
            [batch_size, height_4_d, width_4_d, block_out_channels[2]],
            name="down_block_residual_8",
            is_input=True,
        )
        down_block_residual_9 = Tensor(
            [batch_size, height_8_d, width_8_d, block_out_channels[2]],
            name="down_block_residual_9",
            is_input=True,
        )
        down_block_residual_10 = Tensor(
            [batch_size, height_8_d, width_8_d, block_out_channels[3]],
            name="down_block_residual_10",
            is_input=True,
        )
        down_block_residual_11 = Tensor(
            [batch_size, height_8_d, width_8_d, block_out_channels[3]],
            name="down_block_residual_11",
            is_input=True,
        )
        mid_block_residual = Tensor(
            [batch_size, height_8_d, width_8_d, block_out_channels[3]],
            name="mid_block_residual",
            is_input=True,
        )

    Y = ait_mod(
        sample=latent_model_input_ait,
        timesteps=timesteps_ait,
        encoder_hidden_states=text_embeddings_pt_ait,
        down_block_residual_0=down_block_residual_0,
        down_block_residual_1=down_block_residual_1,
        down_block_residual_2=down_block_residual_2,
        down_block_residual_3=down_block_residual_3,
        down_block_residual_4=down_block_residual_4,
        down_block_residual_5=down_block_residual_5,
        down_block_residual_6=down_block_residual_6,
        down_block_residual_7=down_block_residual_7,
        down_block_residual_8=down_block_residual_8,
        down_block_residual_9=down_block_residual_9,
        down_block_residual_10=down_block_residual_10,
        down_block_residual_11=down_block_residual_11,
        mid_block_residual=mid_block_residual,
        class_labels=class_labels,
        add_embeds=add_embeds,
    )
    shape = [d._attrs["values"] for d in Y._attrs["shape"]]
    print(f'AIT {Y._attrs["name"]} shape: {shape}')
    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    dll_name = model_name + ".dll" if sys.platform == "win32" else model_name + ".so"
    compile_model(
        Y,
        target,
        work_dir,
        model_name,
        constants=params_ait if constants else None,
        dll_name=dll_name,
    )
