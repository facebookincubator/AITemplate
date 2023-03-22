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
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code


class PTSimpleModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden, 4 * hidden)
        self.act1 = torch.nn.functional.gelu
        self.dense2 = torch.nn.Linear(4 * hidden, hidden)
        self.layernorm = torch.nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.act1(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = hidden_states + input
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class AITSimpleModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden, 4 * hidden, specialization="fast_gelu")
        self.dense2 = nn.Linear(4 * hidden, hidden)
        self.layernorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.dense2(hidden_states)
        hidden_states = hidden_states + input
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params


def verify_simple_model(batch_size=1024, hidden=512):
    # create pt model
    pt_model = PTSimpleModel(hidden).cuda().half()

    # create pt input
    x = torch.randn([batch_size, hidden]).cuda().half()

    # run pt model
    pt_model.eval()
    y_pt = pt_model(x)

    # create ait model
    ait_model = AITSimpleModel(hidden)
    X = Tensor(
        shape=[batch_size, hidden],
        name="X",
        dtype="float16",
        is_input=True,
    )
    Y = ait_model(X)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    # map pt weights to ait
    weights = map_pt_params(ait_model, pt_model)

    # code gen
    target = detect_target()
    with compile_model(
        Y, target, "./tmp", "simple_model_demo", constants=weights
    ) as module:
        # create storage for output tensor
        y = torch.empty([batch_size, hidden]).cuda().half()

        # inputs and outputs dict
        inputs = {"X": x}
        outputs = {"Y": y}

        # run
        module.run_with_tensors(inputs, outputs, graph_mode=True)

        # verify output is correct
        print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

        # benchmark ait and pt
        count = 1000
        ait_t, _, _ = module.benchmark_with_tensors(
            inputs, outputs, graph_mode=True, count=count
        )
        print(f"AITemplate time: {ait_t} ms/iter")

        pt_t = benchmark_torch_function(count, pt_model.forward, x)
        print(f"PyTorch eager time: {pt_t} ms/iter")

        # check out the fused graph
        # there are only fused ops in the final graph
        # gemm_rcr_bias_fast_gelu, gemm_rcr_bias_add, and layernorm
        graph = module.debug_sorted_graph
        print("Final graph:")
        print(sorted_graph_pseudo_code(graph))


verify_simple_model()
