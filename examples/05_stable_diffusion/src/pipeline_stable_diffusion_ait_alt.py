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
import inspect
import os
import re
import PIL
import torch

import numpy as np

from typing import List, Optional, Union
from aitemplate.compiler import Model
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.pil_utils import numpy_to_pil
from tqdm import tqdm
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from .compile_lib.compile_vae_alt import map_vae_params
from .modeling.vae import AutoencoderKL as ait_AutoencoderKL
from .pipeline_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)


def preprocess(image, width=512, height=512):
    width, height = map(lambda x: x - x % 32, (width, height))  # resize to integer multiple of 32
    image = image.resize((width, height), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


textenc_conversion_lst = [
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    (
        "token_embedding.weight",
        "transformer.text_model.embeddings.token_embedding.weight",
    ),
    (
        "positional_embedding",
        "transformer.text_model.embeddings.position_embedding.weight",
    ),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def convert_text_enc_state_dict(state_dict):
    if "transformer.resblocks.22.ln_1.bias" not in state_dict.keys():
        return state_dict  # SD1.x
    new_state_dict = {}
    d_model = 1024
    for key, arr in state_dict.items():
        if "resblocks.23" in key:
            continue  # diffusers skips the last layer
        if key in textenc_conversion_map:
            new_state_dict[textenc_conversion_map[key]] = arr
        if key.startswith("transformer."):
            new_key = key[len("transformer."):]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                new_state_dict[new_key + ".q_proj.weight"] = arr[:d_model, :]
                new_state_dict[new_key + ".k_proj.weight"] = arr[
                                                             d_model: d_model * 2, :
                                                             ]
                new_state_dict[new_key + ".v_proj.weight"] = arr[d_model * 2:, :]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                new_state_dict[new_key + ".q_proj.bias"] = arr[:d_model]
                new_state_dict[new_key + ".k_proj.bias"] = arr[d_model: d_model * 2]
                new_state_dict[new_key + ".v_proj.bias"] = arr[d_model * 2:]
            else:
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                new_state_dict[new_key] = arr
    return new_state_dict


# =========================#
#    AITemplate mapping   #
# =========================#
def map_unet_state_dict(state_dict, dim=320):
    params_ait = {}
    for key, arr in state_dict.items():
        arr = arr.to("cuda", dtype=torch.float16)
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            # print("ff.net.0.proj.weight")
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            # print("ff.net.0.proj.bias")
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def map_clip_state_dict(state_dict):
    params_ait = {}
    for key, arr in state_dict.items():
        arr = arr.to("cuda", dtype=torch.float16)
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr

    return params_ait


class StableDiffusionAITPipeline:
    def __init__(self, hf_hub_or_path, ckpt, workdir="tmp/"):
        self.device = torch.device("cuda")
        state_dict = None
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location="cpu")
            while "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            clip_state_dict = {}
            unet_state_dict = {}
            vae_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("cond_stage_model.transformer."):
                    new_key = key.replace("cond_stage_model.transformer.", "")
                    clip_state_dict[new_key] = state_dict[key]
                elif key.startswith("cond_stage_model.model."):
                    new_key = key.replace("cond_stage_model.model.", "")
                    clip_state_dict[new_key] = state_dict[key]
                elif key.startswith("first_stage_model."):
                    new_key = key.replace("first_stage_model.", "")
                    vae_state_dict[new_key] = state_dict[key]
                elif key.startswith("model.diffusion_model."):
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_state_dict[new_key] = state_dict[key]
            clip_state_dict = convert_text_enc_state_dict(clip_state_dict)
            unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict)
            vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict)
            state_dict = None
        self.clip_ait_exe = self.init_ait_module(
            model_name="CLIPTextModel", workdir=workdir
        )
        print("Loading PyTorch CLIP")
        if ckpt is None:
            self.clip_pt = CLIPTextModel.from_pretrained(
                hf_hub_or_path,
                subfolder="text_encoder",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
        else:
            config = CLIPTextConfig.from_pretrained(
                hf_hub_or_path, subfolder="text_encoder"
            )
            self.clip_pt = CLIPTextModel(config)
            clip_state_dict[
                "text_model.embeddings.position_ids"
            ] = self.clip_pt.text_model.embeddings.get_buffer("position_ids")
            self.clip_pt.load_state_dict(clip_state_dict)
        clip_params_ait = map_clip_state_dict(dict(self.clip_pt.named_parameters()))
        print("Setting constants")
        self.clip_ait_exe.set_many_constants_with_tensors(clip_params_ait)
        print("Folding constants")
        self.clip_ait_exe.fold_constants()
        # cleanup
        self.clip_pt = None
        clip_params_ait = None

        self.unet_ait_exe = self.init_ait_module(
            model_name="UNet2DConditionModel", workdir=workdir
        )

        print("Loading PyTorch UNet")
        if ckpt is None:
            self.unet_pt = UNet2DConditionModel.from_pretrained(
                hf_hub_or_path,
                subfolder="unet",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
            self.unet_pt = self.unet_pt.state_dict()
        else:
            self.unet_pt = unet_state_dict
        unet_params_ait = map_unet_state_dict(self.unet_pt)
        print("Setting constants")
        self.unet_ait_exe.set_many_constants_with_tensors(unet_params_ait)
        print("Folding constants")
        self.unet_ait_exe.fold_constants()
        # cleanup
        self.unet_pt = None
        unet_params_ait = None

        self.vae_ait_exe = self.init_ait_module(
            model_name="AutoencoderKL", workdir=workdir
        )
        print("Loading PyTorch VAE")
        if ckpt is None:
            self.vae = AutoencoderKL.from_pretrained(
                hf_hub_or_path,
                subfolder="vae",
                revision="fp16",
                torch_dtype=torch.float16,
            ).cuda()
        else:
            self.vae = dict(vae_state_dict)
        in_channels = 3
        out_channels = 3
        down_block_types = [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ]
        up_block_types = [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ]
        block_out_channels = [128, 256, 512, 512]
        layers_per_block = 2
        act_fn = "silu"
        latent_channels = 4
        sample_size = 512

        ait_vae = ait_AutoencoderKL(
            1,
            64,
            64,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
        )
        print("Mapping parameters...")
        vae_params_ait = map_vae_params(ait_vae, self.vae)
        print("Setting constants")
        self.vae_ait_exe.set_many_constants_with_tensors(vae_params_ait)
        print("Folding constants")
        self.vae_ait_exe.fold_constants()
        # cleanup
        del ait_vae
        del vae_params_ait

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            hf_hub_or_path, subfolder="scheduler"
        )
        # self.scheduler = PNDMScheduler.from_pretrained(
        #     hf_hub_or_path, subfolder="scheduler"
        # )
        self.batch = 1

    def init_ait_module(
            self,
            model_name,
            workdir,
    ):
        mod = Model(os.path.join(workdir, model_name, "test.so"))
        return mod

    def unet_inference(
            self, latent_model_input, timesteps, encoder_hidden_states, height, width
    ):
        exe_module = self.unet_ait_exe
        timesteps_pt = timesteps.expand(self.batch * 2)
        inputs = {
            "input0": latent_model_input.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch * 2
            shape[1] = height // 8
            shape[2] = width // 8
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=77):
        exe_module = self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input, height, width):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            shape[1] = height
            shape[2] = width
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out

    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined  as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        self.batch = batch_size

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))
        # pytorch equivalent
        # text_embeddings = self.clip_pt(text_input.input_ids.to(self.device)).last_hidden_state

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = self.device
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet_inference(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                height=height,
                width=width,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_inference(latents, height, width)
        # pytorch equivalent
        # image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        has_nsfw_concept = None

        if output_type == "pil":
            image = numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Union[str, List[str]],
            init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            latents: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`):
                The negative prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "generator": generator,
            "latents": latents,
            "output_type": output_type,
            "return_dict": return_dict
        }
        if init_image is not None:
            args = {
                "prompt": prompt,
                "strength": strength,
                "init_image": init_image,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "eta": eta,
                "generator": generator,
                "output_type": output_type,
                "return_dict": return_dict
            }
            return self.img2img(**args)
        return self.generate(**args)

    @torch.no_grad()
    def img2img(
            self,
            prompt: Union[str, List[str]],
            init_image: Union[torch.FloatTensor, PIL.Image.Image],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            negative_prompt: Optional[Union[str, List[str]]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        self.batch = batch_size

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = offset

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image, width=width, height=height)

        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(init_image.to(self.device)).latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # expand init_latents for batch_size
        init_latents = torch.cat([init_latents] * batch_size)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, self.device)
        latent_timestep = timesteps[:1].repeat(batch_size)
        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep).to(
            self.device
        )

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [negative_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        for i, t in enumerate(tqdm(timesteps)):
            t_index = i

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
                latent_model_input = latent_model_input.to(self.unet.dtype)
                t = t.to(self.unet.dtype)

            # predict the noise residual
            noise_pred = self.unet_inference(
                latent_model_input, t, encoder_hidden_states=text_embeddings, height=height, width=width
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, t_index, latents, **extra_step_kwargs
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_inference(latents, width=width, height=height)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        has_nsfw_concept = None

        if output_type == "pil":
            image = numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start
