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
import itertools
import unittest
from typing import List

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils, torch_utils


def build_ait_module(
    *,
    batch_sizes,
    input_nonbatch_shape,
    start_indices,
    end_indices,
    n_normalize_over_last_dims,
    gamma_is_none,
    beta_is_none,
    fuse_sigmoid_mul,
    eps,
    test_id,
    ait_dtype="float16",
    workdir="./tmp",
    test_name="strided_layernorm",
):
    target = detect_target()
    X0 = Tensor(
        shape=[
            shape_utils.gen_int_var_min_max(values=batch_sizes, name="input_batch"),
            *input_nonbatch_shape,
        ],
        dtype=ait_dtype,
        name="input",
        is_input=True,
    )
    X1 = ops.dynamic_slice()(X0, start_indices, end_indices)
    layernorm_weight_shape = X1.shape()[-n_normalize_over_last_dims:]
    if gamma_is_none:
        X2 = None
    else:
        X2 = Tensor(
            shape=layernorm_weight_shape,
            dtype=ait_dtype,
            name="gamma",
            is_input=True,
        )
    if beta_is_none:
        X3 = None
    else:
        X3 = Tensor(
            shape=layernorm_weight_shape,
            dtype=ait_dtype,
            name="beta",
            is_input=True,
        )
    if fuse_sigmoid_mul:
        layernorm_op = ops.layernorm()
        sigmoid_op = ops.elementwise(FuncEnum.SIGMOID)
        mul_op = ops.elementwise(FuncEnum.MUL)
        layernorm_out = layernorm_op(X1, X2, X3, layernorm_weight_shape, eps=eps)
        sigmoid_out = sigmoid_op(layernorm_out)
        _ = mul_op(sigmoid_out, X1)
        fused_op = ops.layernorm_sigmoid_mul(layernorm_op, sigmoid_op, mul_op)
        output = fused_op()
    else:
        output = ops.layernorm()(X1, X2, X3, layernorm_weight_shape, eps)
    output._attrs["is_output"] = True
    output._attrs["name"] = "output"
    dll_name = f"test_{test_id}.so"
    return compile_model(
        output,
        target,
        workdir,
        test_name,
        dll_name=dll_name,
    )


def eval_pt(
    *,
    batch_size,
    input_nonbatch_shape,
    start_indices,
    end_indices,
    n_normalize_over_last_dims,
    gamma_is_none,
    beta_is_none,
    fuse_sigmoid_mul,
    eps,
    dtype=torch.float16,
    device="cuda",
):
    dtype_device = {"dtype": dtype, "device": device}
    X0 = torch.randn(batch_size, *input_nonbatch_shape, **dtype_device)
    X1 = X0[[slice(i, j) for i, j in zip(start_indices, end_indices)]]
    layernorm_weight_shape = X1.shape[-n_normalize_over_last_dims:]
    if gamma_is_none:
        X2 = None
    else:
        X2 = torch.randn(layernorm_weight_shape, **dtype_device)
    if beta_is_none:
        X3 = None
    else:
        X3 = torch.randn(layernorm_weight_shape, **dtype_device)
    X4 = torch.nn.functional.layer_norm(
        input=X1,
        normalized_shape=layernorm_weight_shape,
        weight=X2,
        bias=X3,
        eps=eps,
    )
    if fuse_sigmoid_mul:
        output = torch.mul(X1, torch.sigmoid(X4))
    else:
        output = X4
    return {"input": X0, "gamma": X2, "beta": X3, "output": output}


class SliceLayerNormTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceLayerNormTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_slice_layer_norm(
        self,
        *,
        input_nonbatch_shape: List[int] = (16, 64, 1024),
        n_normalize_over_last_dims: int = 1,
        batch_sizes=(3, 4, 7, 11, 18),
        gamma_is_none=False,
        beta_is_none=False,
        fuse_sigmoid_mul=False,
        eps=1e-5,
        start_indices: List[int] = (0,),
        end_indices: List[int] = (None,),
        dtype: str = "float16",
    ):
        input_rank = 1 + len(input_nonbatch_shape)
        if 1 == len(start_indices) and len(start_indices) != input_rank:
            start_indices = [start_indices[0]] * input_rank
        if 1 == len(end_indices) and len(end_indices) != input_rank:
            end_indices = [end_indices[0]] * input_rank

        _layernorm_common_params = {
            "input_nonbatch_shape": input_nonbatch_shape,
            "n_normalize_over_last_dims": n_normalize_over_last_dims,
            "gamma_is_none": gamma_is_none,
            "beta_is_none": beta_is_none,
            "fuse_sigmoid_mul": fuse_sigmoid_mul,
            "eps": eps,
            "start_indices": start_indices,
            "end_indices": end_indices,
        }

        ait_module = build_ait_module(
            batch_sizes=batch_sizes,
            **_layernorm_common_params,
            test_id=self._test_id,
            ait_dtype=dtype,
        )
        self._test_id += 1
        pt_dtype = torch_utils.string_to_torch_dtype(dtype)
        for batch_size in batch_sizes:
            pt_tensors = eval_pt(
                batch_size=batch_size, **_layernorm_common_params, dtype=pt_dtype
            )
            ait_inputs = {
                k: v for k, v in pt_tensors.items() if v is not None and k != "output"
            }
            ait_outputs = {"output": torch.empty_like(pt_tensors["output"])}
            ait_module.run_with_tensors(ait_inputs, ait_outputs)

            self.assertTrue(
                torch.allclose(
                    ait_outputs["output"], pt_tensors["output"], atol=1e-3, rtol=1e-3
                )
            )

    def _test_slice_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shape in (
            # (cuda-half4) kernel
            ((0, 0, 0, 4), (None, None, None, 36), (4, 1, 40)),
            # (generic n < 1024) kernel
            ((0, 0, 0, 11), (None, None, None, 13), (4, 1, 15)),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 0, 1), (None, None, None, 1026), (4, 1, 1027)),
        ):
            self._test_slice_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shape=input_nonbatch_shape,
                **kwargs,
            )

    def _test_middle_slice_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shape in (
            # (cuda-half4) kernel
            ((0, 0, 4, 0), (None, None, 36, None), (2, 40, 4)),
            # (generic n < 1024) kernel
            ((0, 0, 11, 0), (None, None, 13, None), (2, 15, 2)),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 1, 0), (None, None, 1026, None), (2, 1027, 2)),
        ):
            self._test_slice_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shape=input_nonbatch_shape,
                **kwargs,
            )

    def test_slice_layer_norm_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (1, 3),
            (True, False),
            (True, False),
        ):
            self._test_slice_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=False,
            )

    def test_middle_slice_layer_norm_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (2, 3),
            (True, False),
            (True, False),
        ):
            self._test_middle_slice_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=False,
            )

    def test_slice_layer_norm_fuse_sigmoid_mul_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (1, 3),
            (True, False),
            (True, False),
        ):
            self._test_slice_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=True,
            )

    def test_middle_slice_layer_norm_fuse_sigmoid_mul_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (2, 3),
            (True, False),
            (True, False),
        ):
            self._test_middle_slice_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=True,
            )

    @unittest.skipIf(
        detect_target().name() != "cuda", "fp32 is only supported in CUDA backend"
    )
    def test_slice_layer_norm_float32(self):
        self._test_slice_layer_norm_kernels(
            n_normalize_over_last_dims=1,
            gamma_is_none=True,
            beta_is_none=True,
            fuse_sigmoid_mul=False,
            dtype="float32",
        )
        self._test_middle_slice_layer_norm_kernels(
            n_normalize_over_last_dims=2,
            gamma_is_none=True,
            beta_is_none=False,
            fuse_sigmoid_mul=False,
            dtype="float32",
        )
        self._test_slice_layer_norm_kernels(
            n_normalize_over_last_dims=3,
            gamma_is_none=False,
            beta_is_none=True,
            fuse_sigmoid_mul=True,
            dtype="float32",
        )
        self._test_middle_slice_layer_norm_kernels(
            n_normalize_over_last_dims=2,
            gamma_is_none=False,
            beta_is_none=False,
            fuse_sigmoid_mul=True,
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
