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
"""
AITemplate Toy Example
======================
Two patterns:
  1. Raw operator graph: elementwise tanh(X + 3)
  2. nn.Module style: Linear + GeLU + residual + LayerNorm

Run with:
  buck run fbcode//aitemplate/AITemplate/examples:toy_example
"""

import logging

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing.detect_target import FBCUDA


def _get_target(**kwargs):
    """Create AIT CUDA target using the actual GPU compute capability.

    On virtual hosts /etc/fbwhoami reports MODEL_NAME=VIRTUAL which causes
    detect_target() to default to SM80 even when H100 (SM90) GPUs are present.
    We detect the real SM version via torch.cuda instead.
    """
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_arch = str(cc_major * 10 + cc_minor)
    return FBCUDA(arch=gpu_arch, **kwargs)


class PTSimpleModel(torch.nn.Module):
    """PyTorch reference model."""

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
    """AITemplate equivalent — fuses GEMM + bias + GeLU into one kernel."""

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
    mapped = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped[ait_name] = pt_params[name]
    return mapped


def run_elementwise_example():
    """Example 1: Raw elementwise ops — Y = tanh(X + 3)"""
    print("\n" + "=" * 60)
    print("Example 1: Elementwise ops  (Y = tanh(X + 3))")
    print("=" * 60)

    # 1. Build graph
    X = Tensor(shape=[1024, 256], name="X", dtype="float16", is_input=True)
    Y = ops.tanh(X + 3.0)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"

    # 2. Compile
    target = _get_target()
    logging.getLogger("aitemplate").setLevel(logging.DEBUG)
    module = compile_model(Y, target, "./tmp", "toy_tanh_add")

    # 3. Run inference
    x_pt = torch.randn(1024, 256).cuda().half()
    y_ait = torch.empty(1024, 256).cuda().half()
    module.run_with_tensors({"X": x_pt}, {"Y": y_ait})

    # 4. Verify against PyTorch
    y_pt = torch.tanh(x_pt + 3.0)
    close = torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2)
    assert close, "Elementwise example: results do not match PyTorch!"
    print(f"Results match PyTorch: {close}")


def run_nn_module_example():
    """Example 2: nn.Module with weight mapping."""
    print("\n" + "=" * 60)
    print("Example 2: nn.Module  (Linear + GeLU + residual + LayerNorm)")
    print("=" * 60)

    batch_size, hidden = 1024, 512

    # 1. Create and run PyTorch model
    pt_model = PTSimpleModel(hidden).cuda().half()
    pt_model.eval()
    x = torch.randn(batch_size, hidden).cuda().half()
    y_pt = pt_model(x)

    # 2. Build AIT graph
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

    # 3. Map weights and compile
    weights = map_pt_params(ait_model, pt_model)
    target = _get_target()
    logging.getLogger("aitemplate").setLevel(logging.DEBUG)
    with compile_model(
        Y, target, "./tmp", "toy_simple_model", constants=weights
    ) as module:
        # 4. Run inference
        y_ait = torch.empty(batch_size, hidden).cuda().half()
        module.run_with_tensors({"X": x}, {"Y": y_ait})

        # 5. Verify
        close = torch.allclose(y_ait, y_pt, atol=1e-2, rtol=1e-2)
        assert close, "nn.Module example: results do not match PyTorch!"
        print(f"Results match PyTorch: {close}")


def main():
    run_elementwise_example()
    run_nn_module_example()
    print("\nAll examples passed!")


if __name__ == "__main__":
    main()
