from typing import Optional

import numpy as np
import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor

from aitemplate.testing import detect_target
from aitemplate.utils.import_path import import_parent
from torch import nn

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.modeling.clip import CLIPTextEmbeddings as ait_CLIPTextEmbeddings


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][-1] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))


class pt_CLIPTextEmbeddings(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=49408, max_position_embeddings=77):
        super().__init__()
        embed_dim = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


def verify_CLIPTextEmbeddings(
    batch_size=1,
    seqlen=16,
    hidden_size=768,
    vocab_size=49408,
    max_position_embeddings=77,
    use_fp16_acc=False,
):
    pt_mod = pt_CLIPTextEmbeddings().cuda().half()
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seqlen), dtype=torch.long
    ).cuda()

    position_ids = torch.arange(seqlen).expand((batch_size, -1)).cuda()

    print("input_ids shape:", input_ids.shape)
    pt_y = pt_mod(input_ids, position_ids)
    print("pt output:", pt_y.shape)

    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        print(key, arr.shape)
        params_ait[key.replace(".", "_")] = arr

    # params_ait["position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)).cuda()
    ait_mod = ait_CLIPTextEmbeddings()
    ait_mod.name_parameter_tensor()

    batch_size_d = IntVar(values=[1, 8], name="batch_size")

    inputs_ait = (
        Tensor([batch_size_d, seqlen], name="input0", dtype="int64", is_input=True),
        Tensor([batch_size_d, seqlen], name="input1", dtype="int64", is_input=True),
    )

    Y = ait_mod(*inputs_ait)
    mark_output(Y)
    # return
    ait_param_names = [x[0] for x in ait_mod.named_parameters()]
    # print(ait_param_names)

    target = detect_target(use_fp16_acc=use_fp16_acc)
    exe_module = compile_model(Y, target, "./tmp", "clip_embeding")
    for name, weight in params_ait.items():
        exe_module.set_constant_with_tensor(name, weight)

    inputs = [input_ids, position_ids]
    # y = torch.empty(get_int_shape(Y)).cuda().half()
    ys = []
    num_ouputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_ouputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch_size
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys)
    # exe_module.benchmark_with_tensors(
    #     inputs, outputs, count=100, repeat=1
    # )

    eps = 1e-2
    np.testing.assert_allclose(
        pt_y.detach().cpu().numpy(),
        ys[0].cpu().numpy(),
        atol=eps,
        rtol=eps,
    )
    print("clip_embeding verification pass")


verify_CLIPTextEmbeddings(batch_size=4)
