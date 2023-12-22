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
from typing import Optional, Tuple, Union

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import get_down_block, get_up_block, UNetMidBlock2DCrossAttn


class UNet2DConditionModel(nn.Module):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int`, *optional*): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        use_linear_projection (`bool`, *optional*, defaults to False): Use linear projection instead of 1x1 convolution.
    """

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        only_cross_attention=[True, True, True, False],
        conv_in_kernel=3,
        dtype="float16",
        time_embedding_dim=None,
        projection_class_embeddings_input_dim=None,
        addition_embed_type=None,
        transformer_layers_per_block=[1, 1, 1, 1],
    ):
        super().__init__()
        self.center_input_sample = center_input_sample
        self.sample_size = sample_size
        self.time_embedding_dim = time_embedding_dim
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # input
        self.in_channels = in_channels
        if self.in_channels % 4 != 0:
            in_channels = self.in_channels + (4 - (self.in_channels % 4))
        else:
            in_channels = self.in_channels
        conv_in_padding = (conv_in_kernel - 1) // 2
        print("in_channels", in_channels)
        if in_channels < 8 and detect_target().name() == "cuda":
            self.conv_in = nn.Conv2dBiasFewChannels(
                in_channels, block_out_channels[0], 3, 1, conv_in_padding, dtype=dtype
            )
        else:
            self.conv_in = nn.Conv2dBias(
                in_channels, block_out_channels[0], 3, 1, conv_in_padding, dtype=dtype
            )
        # time
        self.time_proj = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos,
            freq_shift,
            dtype=dtype,
            arange_name="arange",
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, dtype=dtype
        )
        self.class_embed_type = class_embed_type
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(
                [num_class_embeds, time_embed_dim], dtype=dtype
            )
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, dtype=dtype
            )
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(dtype=dtype)
        else:
            self.class_embedding = None

        if addition_embed_type == "text_time":
            self.add_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim, dtype=dtype
            )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            dtype=dtype,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block)
        )
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=reversed_attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                dtype=dtype,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=norm_eps,
            use_swish=True,
            dtype=dtype,
        )

        self.conv_out = nn.Conv2dBias(
            block_out_channels[0], out_channels, 3, 1, 1, dtype=dtype
        )

    def forward(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        down_block_residual_0=None,
        down_block_residual_1=None,
        down_block_residual_2=None,
        down_block_residual_3=None,
        down_block_residual_4=None,
        down_block_residual_5=None,
        down_block_residual_6=None,
        down_block_residual_7=None,
        down_block_residual_8=None,
        down_block_residual_9=None,
        down_block_residual_10=None,
        down_block_residual_11=None,
        mid_block_residual=None,
        class_labels: Optional[Tensor] = None,
        add_embeds: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        """r
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        down_block_additional_residuals = (
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
        )
        mid_block_additional_residual = mid_block_residual
        if down_block_additional_residuals[0] is None:
            down_block_additional_residuals = None

        # 1. time
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = ops.batch_gather()(
                self.class_embedding.weight.tensor(), class_labels
            )
            emb = emb + class_emb

        if add_embeds is not None:
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        # 2. pre-process
        if self.in_channels % 4 != 0:
            channel_pad = self.in_channels + (4 - (self.in_channels % 4))
            sample = ops.pad_last_dim(4, channel_pad)(sample)

        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
            # return sample

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_additional_residual._attrs[
                    "shape"
                ] = down_block_res_sample._attrs["shape"]
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        if mid_block_additional_residual is not None:
            mid_block_additional_residual._attrs["shape"] = sample._attrs["shape"]
            sample += mid_block_additional_residual
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "attentions")
                and upsample_block.attentions is not None
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)
        sample._attrs["is_output"] = True
        sample._attrs["name"] = "latent_output"
        return sample
