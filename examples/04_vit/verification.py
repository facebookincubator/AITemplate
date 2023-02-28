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
import click
import numpy as np
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from modeling.vision_transformer import VisionTransformer
from timm.models.vision_transformer import vit_base_patch16_224, vit_large_patch16_384

from weight_utils import export_to_torch_tensor


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


USE_CUDA = detect_target().name() == "cuda"


def compile_vit(
    batch_size=128,
    img_size=224,
    patch_size=16,
    embed_dim=768,
    num_heads=12,
    depth=12,
    class_token=True,
    global_pool="token",
    use_fp16_acc=True,
):
    seqlen = (img_size // patch_size) ** 2 + (1 if class_token else 0)
    ait_model = VisionTransformer(
        batch_size=batch_size,
        img_size=img_size,
        class_token=class_token,
        global_pool=global_pool,
        num_heads=num_heads,
        embed_dim=embed_dim,
        patch_size=patch_size,
        depth=depth,
        act_layer="GELU",
    )
    ait_model.name_parameter_tensor()
    inputs_ait = Tensor(
        [batch_size, img_size, img_size, 3], name="input0", is_input=True
    )
    Y = ait_model(inputs_ait)
    mark_output(Y)

    target = detect_target(use_fp16_acc=use_fp16_acc)
    exe_module = compile_model(
        Y, target, "./tmp", "vision_transformer_bs%d_seq%d" % (batch_size, seqlen)
    )
    return exe_module


def verification(
    model_name,
    batch_size=3,
    use_fp16_acc=True,
):
    img_size = 224
    embed_dim = 768
    depth = 12
    patch_size = 16
    num_heads = 12
    class_token = True
    global_pool = "token"
    if model_name == "vit_base_patch16_224":
        img_size = 224
        embed_dim = 768
        depth = 12
        patch_size = 16
        num_heads = 12
        pt_mod = vit_base_patch16_224(pretrained=True).cuda().half()
    elif model_name == "vit_large_patch16_384":
        img_size = 384
        embed_dim = 1024
        depth = 24
        patch_size = 16
        num_heads = 16
        pt_mod = vit_large_patch16_384(pretrained=True).cuda().half()

    seqlen = (img_size // patch_size) ** 2 + (1 if class_token else 0)
    input_pt = torch.randn([batch_size, 3, img_size, img_size]).cuda().half() * 255
    pt_ys = pt_mod(input_pt)
    pt_ys = pt_ys.reshape((batch_size, 1, -1))

    ait_mod = compile_vit(
        batch_size=batch_size,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        class_token=True,
        global_pool=global_pool,
        use_fp16_acc=use_fp16_acc,
    )

    # convert weights
    params_ait = export_to_torch_tensor(model_name, True)
    params_ait["cls_token_mask"] = torch.zeros((batch_size, 1, embed_dim)).cuda().half()
    if detect_target().name() == "cuda":
        ait_key = "attn_cu_length"
        for i in range(depth):
            prefix = "blocks_%d" % (i)
            cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
            params_ait[f"{prefix}_{ait_key}"] = torch.from_numpy(cu_len).cuda()

    # set weights
    ait_mod.set_many_constants_with_tensors(params_ait)
    ait_mod.fold_constants(sync=True)

    inputs = [input_pt.permute((0, 2, 3, 1)).contiguous()]
    ys = []
    num_outputs = len(ait_mod.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = ait_mod.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    ait_mod.run_with_tensors(inputs, ys)
    eps = 1e-1
    np.testing.assert_allclose(
        pt_ys.detach().cpu().numpy(),
        ys[0].cpu().numpy(),
        atol=eps,
        rtol=eps,
    )
    print("vision transformer verification pass")


@click.command()
@click.option("--model-name", type=str, default="vit_base_patch16_224")
@click.option("--use-fp16-acc", type=bool, default=True)
def main(model_name, use_fp16_acc):
    if model_name not in ("vit_base_patch16_224", "vit_large_patch16_384"):
        raise ValueError(
            "model name should be vit_base_patch16_224 or vit_large_patch16_384"
        )
    verification(model_name, use_fp16_acc=use_fp16_acc)


if __name__ == "__main__":
    main()
