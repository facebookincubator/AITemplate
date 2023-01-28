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
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils, torch_utils


def build_ait_module(
    *,
    batch_sizes,
    eps,
    test_id,
    ait_dtype="float16",
    workdir="./tmp",
    test_name="strided_layernorm_reshape",
):
    input_nonbatch_shape = [6912]
    target = detect_target()
    batch_size = shape_utils.gen_int_var_min_max(values=batch_sizes, name="input_batch")
    inputs = Tensor(
        shape=[
            batch_size,
            *input_nonbatch_shape,
        ],
        dtype=ait_dtype,
        name="input",
        is_input=True,
    )
    slice_out = ops.dynamic_slice()(
        inputs, start_indices=[0, 0], end_indices=[None, 6400]
    )
    reshape_out = ops.reshape()(slice_out, shape=[-1, 128, 50])
    layernorm_weight_shape = reshape_out.shape()[-2:]
    gamma_beta_params = {
        "shape": layernorm_weight_shape,
        "dtype": ait_dtype,
        "is_input": True,
    }
    gammas = Tensor(
        name="gamma",
        **gamma_beta_params,
    )
    betas = Tensor(
        name="beta",
        **gamma_beta_params,
    )
    layernorm_out = ops.layernorm()(
        reshape_out, gammas, betas, layernorm_weight_shape, eps
    )
    output = ops.reshape()(layernorm_out, shape=[-1, 6400])

    output._attrs["is_output"] = True
    output._attrs["name"] = "output"
    dll_name = f"test_{test_id}.so"
    return (
        inputs,
        output,
        compile_model(output, target, workdir, test_name, dll_name=dll_name),
    )


def eval_pt(
    *,
    batch_size,
    eps,
    dtype=torch.float16,
    device="cuda",
):
    dtype_device = {"dtype": dtype, "device": device}
    inputs = torch.randn(batch_size, 6912, **dtype_device)
    slice_out = inputs[:, :6400]
    reshape_out = slice_out.reshape(-1, 128, 50)
    layernorm_weight_shape = reshape_out.shape[-2:]
    gammas = torch.randn(layernorm_weight_shape, **dtype_device)
    betas = torch.randn(layernorm_weight_shape, **dtype_device)
    layernorm_out = torch.nn.functional.layer_norm(
        input=reshape_out,
        normalized_shape=layernorm_weight_shape,
        weight=gammas,
        bias=betas,
        eps=eps,
    )
    output = layernorm_out.reshape(shape=[-1, 6400])

    return {"input": inputs, "gamma": gammas, "beta": betas, "output": output}


class SliceLayerNormReshapeTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceLayerNormReshapeTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_slice_layer_norm_reshape(
        self,
        *,
        dtype="float16",
        batch_sizes=(3, 4),
        eps=1e-5,
        atol=1e-3,
        rtol=1e-3,
    ):
        ait_in_node, ait_out_node, ait_module = build_ait_module(
            batch_sizes=batch_sizes,
            eps=eps,
            test_id=self._test_id,
            ait_dtype=dtype,
        )
        self._test_id += 1

        for op_name in (
            next(iter(ait_in_node._attrs["dst_ops"]))._attrs["name"],
            next(iter(ait_out_node._attrs["src_ops"]))._attrs["name"],
        ):
            self.assertRegex(op_name, "layernorm")

        pt_dtype = torch_utils.string_to_torch_dtype(dtype)
        for batch_size in batch_sizes:
            pt_tensors = eval_pt(batch_size=batch_size, eps=eps, dtype=pt_dtype)
            ait_inputs = {k: v for k, v in pt_tensors.items() if k != "output"}
            ait_outputs = {"output": torch.empty_like(pt_tensors["output"])}
            ait_module.run_with_tensors(ait_inputs, ait_outputs)

            self.assertTrue(
                torch.allclose(
                    ait_outputs["output"], pt_tensors["output"], atol=atol, rtol=rtol
                )
            )

    def test_slice_layer_norm_reshape_float16(self):
        self._test_slice_layer_norm_reshape()

    @unittest.skipIf(
        detect_target().name() != "cuda", "fp32 is only supported in CUDA backend"
    )
    def test_slice_layer_norm_reshape_float32(self):
        self._test_slice_layer_norm_reshape(dtype="float32")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
