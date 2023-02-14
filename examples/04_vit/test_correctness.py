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

import io
import unittest

import numpy as np
import torch
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import Tensor

from aitemplate.testing import detect_target

try:
    from libfb.py.asyncio.await_utils import await_sync
    from manifold.clients.python import ManifoldClient
except ImportError:
    ManifoldClient = None

from parameterized import parameterized

from timm.models.vision_transformer import vit_base_patch16_224, vit_large_patch16_384

from .modeling.vision_transformer import VisionTransformer


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


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


class VITVerification(unittest.TestCase):
    @parameterized.expand(["vit_base_patch16_224", "vit_large_patch16_384"])
    def test_vit(self, model_name):
        if model_name == "vit_base_patch16_224":
            img_size = 224
            depth = 12
            embed_dim = 768
            num_heads = 12
            global_pool = "token"
            vit_pt_def = vit_base_patch16_224
            path = "tree/aitemplate/vit-pt/vit_base_patch16_224.pt"

        elif model_name == "vit_large_patch16_384":
            img_size = 384
            depth = 24
            embed_dim = 1024
            num_heads = 16
            vit_pt_def = vit_large_patch16_384
            path = "tree/aitemplate/vit-pt/vit_large_patch16_384.pt"
        if ManifoldClient is None:
            vit_pt = vit_pt_def(pretrained=True)
        else:
            stream = io.BytesIO()
            with ManifoldClient.get_client(bucket="glow_test_data") as client:
                await_sync(
                    client.get(
                        path,
                        stream,
                    )
                )
            stream.seek(0)
            vit_pt = vit_pt_def(pretrained=False)
            vit_pt.load_state_dict(torch.load(stream))
        global_pool = "token"
        patch_size = 16
        vit_pt = vit_pt.cuda().half()
        batch_size = 1
        vit_ait = compile_vit(
            batch_size=batch_size,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            class_token=True,
            global_pool=global_pool,
            use_fp16_acc=False,
        )
        nc = 3
        seqlen = (img_size // patch_size) ** 2 + 1

        # prepare params
        params_pt = vit_pt.named_parameters()
        params_ait = {}
        for key, arr in params_pt:
            ait_key = key.replace(".", "_")
            if len(arr.shape) == 4:
                arr = arr.permute((0, 2, 3, 1)).contiguous()
                if detect_target().name() == "cuda":
                    conv0_w_pad = (
                        torch.zeros((embed_dim, patch_size, patch_size, 4))
                        .cuda()
                        .half()
                    )
                    conv0_w_pad[:, :, :, :3] = arr
                    arr = conv0_w_pad
            params_ait[f"{ait_key}"] = arr
        params_ait["cls_token_mask"] = (
            torch.zeros((batch_size, 1, embed_dim)).cuda().half()
        )
        if detect_target().name() == "cuda":
            ait_key = "attn_cu_length"
            for i in range(depth):
                prefix = "blocks_%d" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[f"{prefix}_{ait_key}"] = torch.from_numpy(cu_len).cuda()

        # set weights
        for name, weight in params_ait.items():
            vit_ait.set_constant_with_tensor(name, weight)

        with torch.no_grad():
            x_pt = (
                torch.rand(
                    (batch_size, nc, img_size, img_size),
                    dtype=torch.float16,
                    device="cuda",
                )
                * 255
            )
            x_ait = x_pt.permute(0, 2, 3, 1).contiguous()
            y_pt = vit_pt(x_pt).reshape(batch_size, 1, -1)
            y_ait = torch.empty_like(y_pt)
            vit_ait.run_with_tensors([x_ait], [y_ait])
            torch.testing.assert_close(y_ait, y_pt, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
