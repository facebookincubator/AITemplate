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
Unittests for FusedLayernormSigmoidMul Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FusedLayernormSigmoidMulTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FusedLayernormSigmoidMulTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_fused_layernorm_sigmoid_mul(
        self,
        MS=(),
        NS=(16,),
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        atol=1e-2,
        rtol=1e-2,
        eps=1e-5,
        dtype="float16",
    ):
        logging.info(
            f"_test_fused_layernorm_sigmoid_mul: M={MS}, N={NS}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}"
            f"dtype={dtype}"
        )
        assert isinstance(MS, (list, tuple))
        assert isinstance(NS, (list, tuple))

        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=[1, 1024]), *MS, *NS],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=NS,
                dtype=dtype,
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=NS,
                dtype=dtype,
                name="beta",
                is_input=True,
            )
        if use_size_op:
            norm_shapes = [
                ops.getitem()(ops.size()(X1), i) for i in range(1 + len(MS), X1._rank())
            ]
        else:
            norm_shapes = [IntImm(n) for n in NS]
        X4 = (
            ops.layernorm()(X1, X2, X3, norm_shapes, eps)
            if not use_size_op
            else ops.layernorm()(X1, X2, X3, norm_shapes, eps)
        )
        X5 = ops.elementwise(FuncEnum.SIGMOID)(X4)
        X6 = ops.elementwise(FuncEnum.MUL)(X1, X5)
        X6._attrs["is_output"] = True
        X6._attrs["name"] = "output"

        target = detect_target()
        with compile_model(
            X6,
            target,
            "./tmp",
            f"fused_layernorm_sigmoid_mul_test_{self._test_id}",
        ) as module:
            self._test_id += 1
            for batch_size in [50, 900, 1024]:
                logging.info(
                    f"Run test layernorm_sigmoid_mul. Problem size {[batch_size,] + list(MS) + list(NS)}"
                )
                x1_pt = get_random_torch_tensor([batch_size, *MS, *NS], dtype=dtype)
                if gamma_is_none:
                    x2_pt = None
                else:
                    x2_pt = get_random_torch_tensor(NS, dtype=dtype)
                if beta_is_none:
                    x3_pt = None
                else:
                    x3_pt = get_random_torch_tensor(NS, dtype=dtype)

                x4_pt = torch.nn.functional.layer_norm(x1_pt, NS, x2_pt, x3_pt, eps=eps)
                x6_pt = torch.mul(x1_pt, torch.sigmoid(x4_pt))

                inputs = {"X": x1_pt}
                if not gamma_is_none:
                    inputs["gamma"] = x2_pt
                if not beta_is_none:
                    inputs["beta"] = x3_pt
                x6 = torch.empty_like(x6_pt)
                module.run_with_tensors(inputs, [x6])
                torch.testing.assert_close(x6, x6_pt, atol=atol, rtol=rtol),

    def test_fused_layernorm_sigmoid_mul_fp16(self):
        for eps in (1e-5, 1e-1):
            # half4 kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1496,),
                eps=eps,
                dtype="float16",
            )
            # block_size = n kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(515,),
                eps=eps,
                dtype="float16",
            )
            # block_size = 512 kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                eps=eps,
                dtype="float16",
            )

        # test ND inputs
        eps = 1e-5
        # half4 kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(64, 8),
            eps=eps,
            dtype="float16",
        )
        # block_size = n kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(213, 2),
            eps=eps,
            dtype="float16",
        )
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(3, 2),
            eps=eps,
            dtype="float16",
        )
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(1, 1),
            eps=eps,
            dtype="float16",
        )
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(0, 1),
            eps=eps,
            dtype="float16",
        )
        # block_size = 512 kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 4),
            NS=(1055, 5),
            eps=eps,
            dtype="float16",
        )

        self._test_fused_layernorm_sigmoid_mul(
            NS=(1496,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="float16",
        )
        self._test_fused_layernorm_sigmoid_mul(
            NS=(515,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="float16",
        )
        for use_size_op in (True, False):
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                use_size_op=use_size_op,
                dtype="float16",
            )
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                gamma_is_none=True,
                beta_is_none=True,
                use_size_op=use_size_op,
                dtype="float16",
            )
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1496,),
                gamma_is_none=True,
                use_size_op=use_size_op,
                dtype="float16",
            )
            self._test_fused_layernorm_sigmoid_mul(
                NS=(515,),
                beta_is_none=True,
                use_size_op=use_size_op,
                dtype="float16",
            )

    def test_fused_layernorm_sigmoid_mul_fp32(self):
        for eps in (1e-5, 1e-1):
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1496,),
                eps=eps,
                dtype="float32",
            )
            # block_size = n kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(515,),
                eps=eps,
                dtype="float32",
            )
            # block_size = 512 kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                eps=eps,
                dtype="float32",
            )

        # test ND inputs
        eps = 1e-5
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(64, 8),
            eps=eps,
            dtype="float32",
        )
        # block_size = n kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(213, 2),
            eps=eps,
            dtype="float32",
        )
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(3, 2),
            eps=eps,
            dtype="float32",
        )
        # block_size = 512 kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 4),
            NS=(1055, 5),
            eps=eps,
            dtype="float32",
        )

        self._test_fused_layernorm_sigmoid_mul(
            NS=(1496,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="float32",
        )
        self._test_fused_layernorm_sigmoid_mul(
            NS=(515,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="float32",
        )

    def test_fused_layernorm_sigmoid_mul_bf16(self):
        for eps in (1e-5, 1e-1):
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1496,),
                eps=eps,
                dtype="bfloat16",
                atol=1e-2,
                rtol=1e-2,
            )
            # block_size = n kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(515,),
                eps=eps,
                dtype="bfloat16",
                atol=1e-2,
                rtol=1e-2,
            )
            # block_size = 512 kernel
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                eps=eps,
                dtype="bfloat16",
                atol=1e-2,
                rtol=1e-2,
            )

        # test ND inputs
        eps = 1e-5
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(64, 8),
            eps=eps,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        # block_size = n kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(213, 2),
            eps=eps,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 2),
            NS=(3, 2),
            eps=eps,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        # block_size = 512 kernel
        self._test_fused_layernorm_sigmoid_mul(
            MS=(2, 4),
            NS=(1055, 5),
            eps=eps,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )

        self._test_fused_layernorm_sigmoid_mul(
            NS=(1496,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        self._test_fused_layernorm_sigmoid_mul(
            NS=(515,),
            gamma_is_none=True,
            beta_is_none=True,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )

    # dim0 is batch size
    def _test_batch_fused_layernorm_sigmoid_mul(
        self,
        M,
        N,
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
        atol=1e-2,
        rtol=1e-2,
        dtype="float16",
    ):
        logging.info(
            f"_test_batch_fused_layernorm_sigmoid_mul: M={M}, N={N}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}"
        )
        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(M), IntImm(N)],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(N)],
                dtype=dtype,
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(N)],
                dtype=dtype,
                name="beta",
                is_input=True,
            )
        X4 = (
            ops.batch_layernorm_sigmoid_mul()(X1, X2, X3, [IntImm(N)], eps)
            if not use_size_op
            else ops.batch_layernorm_sigmoid_mul()(
                X1, X2, X3, [ops.getitem()(ops.size()(X1), 2)], eps
            )
        )
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        with compile_model(
            X4,
            target,
            "./tmp",
            f"batch_fused_layernorm_sigmoid_mul_{M}_{N}_test_{self._test_id}",
        ) as module:
            self._test_id += 1
            for batch_size in [2, 16, 32]:
                logging.info(
                    f"Run test batch_layernorm_sigmoid_mul. Problem size [{batch_size}, {M}, {N}]"
                )
                xs_pt = [
                    get_random_torch_tensor([M, N], dtype=dtype)
                    for i in range(batch_size)
                ]
                if gamma_is_none:
                    gammas_pt = [None] * batch_size
                else:
                    gammas_pt = [
                        get_random_torch_tensor([N], dtype=dtype)
                        for i in range(batch_size)
                    ]
                if beta_is_none:
                    betas_pt = [None] * batch_size
                else:
                    betas_pt = [
                        get_random_torch_tensor([N], dtype=dtype)
                        for i in range(batch_size)
                    ]

                ys_pt = []
                for i in range(batch_size):
                    y0 = torch.nn.functional.layer_norm(
                        xs_pt[i],
                        xs_pt[i].size()[1:],
                        gammas_pt[i],
                        betas_pt[i],
                        eps=eps,
                    )
                    y = torch.mul(xs_pt[i], torch.sigmoid(y0))
                    ys_pt.append(y)
                y_t = torch.stack(ys_pt, dim=0)

                x_pt = torch.stack(xs_pt, dim=0)
                if not gamma_is_none:
                    gamma_pt = torch.stack(gammas_pt, dim=0)
                if not beta_is_none:
                    beta_pt = torch.stack(betas_pt, dim=0)

                inputs = {"X": x_pt}
                if not gamma_is_none:
                    inputs["gamma"] = gamma_pt
                if not beta_is_none:
                    inputs["beta"] = beta_pt
                x4 = torch.empty_like(y_t)
                module.run_with_tensors(inputs, [x4])
                torch.testing.assert_close(x4, y_t, atol=atol, rtol=rtol)

    # dim1 is the batch size
    def _test_batch_fused_layernorm_sigmoid_mul_dim1(
        self,
        B,
        N,
        gamma_is_none=False,
        beta_is_none=False,
        atol=1e-2,
        rtol=1e-2,
        dtype="float16",
    ):
        logging.info(
            f"_test_batch_fused_layernorm_sigmoid_mul_dim1: M={B}, N={N}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}"
        )
        X1 = Tensor(
            shape=[
                IntImm(B),
                IntVar(name="input_batch", values=[128, 1024]),
                IntImm(N),
            ],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=[IntImm(B), IntImm(N)],
                dtype=dtype,
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=[IntImm(B), IntImm(N)],
                dtype=dtype,
                name="beta",
                is_input=True,
            )
        X4 = ops.batch_layernorm_sigmoid_mul(normalized_shape=[IntImm(N)])(X1, X2, X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        with compile_model(
            X4,
            target,
            "./tmp",
            f"batch_fused_layernorm_sigmoid_mul_dim1_{B}_{N}_test_{self._test_id}",
        ) as module:
            self._test_id += 1
            for M in [128, 1024]:
                logging.info(
                    f"Run test batch_layernorm_sigmoid_mul. Problem size [{B}, {M}, {N}]"
                )
                xs_pt = [get_random_torch_tensor([M, N], dtype=dtype) for i in range(B)]
                if gamma_is_none:
                    gammas_pt = [None] * B
                else:
                    gammas_pt = [
                        get_random_torch_tensor([N], dtype=dtype) for i in range(B)
                    ]
                if beta_is_none:
                    betas_pt = [None] * B
                else:
                    betas_pt = [
                        get_random_torch_tensor([N], dtype=dtype) for i in range(B)
                    ]

                ys_pt = []
                for i in range(B):
                    y0 = torch.nn.functional.layer_norm(
                        xs_pt[i], xs_pt[i].size()[1:], gammas_pt[i], betas_pt[i]
                    )
                    y = torch.mul(xs_pt[i], torch.sigmoid(y0))
                    ys_pt.append(y)
                y_t = torch.stack(ys_pt, dim=0)

                x_pt = torch.stack(xs_pt, dim=0)
                if not gamma_is_none:
                    gamma_pt = torch.stack(gammas_pt, dim=0)
                if not beta_is_none:
                    beta_pt = torch.stack(betas_pt, dim=0)

                inputs = {"X": x_pt}
                if not gamma_is_none:
                    inputs["gamma"] = gamma_pt
                if not beta_is_none:
                    inputs["beta"] = beta_pt
                x4 = torch.empty_like(y_t)
                module.run_with_tensors(inputs, [x4])
                torch.testing.assert_close(x4, y_t, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            param("float16"),
            param("float32"),
            param("bfloat16"),
        ]
    )
    def test_batch_fused_layernorm_sigmoid_mul(self, dtype: str):
        for eps in (1e-5, 1e-1):
            self._test_batch_fused_layernorm_sigmoid_mul(
                512,
                1024,
                eps=eps,
                dtype=dtype,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                512,
                64,
                eps=eps,
                dtype=dtype,
            )

        self._test_batch_fused_layernorm_sigmoid_mul(
            512,
            1024,
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )
        self._test_batch_fused_layernorm_sigmoid_mul(
            512,
            64,
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )
        for use_size_op in (True, False):
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024,
                1055,
                use_size_op=use_size_op,
                eps=1e-1,
                dtype=dtype,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024,
                1055,
                use_size_op=use_size_op,
                dtype=dtype,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024,
                1055,
                gamma_is_none=True,
                beta_is_none=True,
                use_size_op=use_size_op,
                dtype=dtype,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                512,
                1024,
                gamma_is_none=True,
                use_size_op=use_size_op,
                dtype=dtype,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                512,
                1024,
                beta_is_none=True,
                use_size_op=use_size_op,
                dtype=dtype,
            )

        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            1,
            512,
            dtype=dtype,
        )
        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            16,
            512,
            dtype=dtype,
        )

        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            1,
            512,
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )
        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            16,
            512,
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )

    def _test_group_fused_layernorm_sigmoid_mul(
        self,
        input_shapes,
        norm_ndim=1,
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
        fuse_sigmoid_mul=True,
        atol=1e-2,
        rtol=1e-2,
        dtype="float16",
    ):
        testname = (
            f"group_fused_layernorm_sigmoid_mul_test_{dtype}_{self._test_id}"
            if fuse_sigmoid_mul
            else f"group_layernorm_test_{dtype}_{self._test_id}"
        )
        self._test_id += 1
        logging.info(
            f"{testname}: input_shapes={input_shapes}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}, "
            f"use_size_op={use_size_op}, dtype={dtype}"
        )
        inputs = []
        gammas = []
        betas = []
        normalized_shapes = []
        batch_ndim = len(input_shapes[0]) - norm_ndim
        for i, shape in enumerate(input_shapes):
            inputs.append(
                Tensor(
                    shape=[IntImm(n) for n in shape],
                    dtype=dtype,
                    name="X_" + str(i),
                    is_input=True,
                )
            )
            gamma = (
                None
                if gamma_is_none
                else Tensor(
                    shape=[IntImm(n) for n in shape[batch_ndim:]],
                    dtype=dtype,
                    name="gamma_" + str(i),
                    is_input=True,
                )
            )
            gammas.append(gamma)
            beta = (
                None
                if beta_is_none
                else Tensor(
                    shape=[IntImm(n) for n in shape[batch_ndim:]],
                    dtype=dtype,
                    name="beta_" + str(i),
                    is_input=True,
                )
            )
            betas.append(beta)
            if use_size_op:
                normalized_shapes.append(
                    [
                        ops.getitem()(ops.size()(inputs[-1]), i)
                        for i in range(batch_ndim, len(shape))
                    ]
                )
            else:
                normalized_shapes.append([IntImm(n) for n in shape[batch_ndim:]])

        if fuse_sigmoid_mul:
            Ys = ops.group_layernorm_sigmoid_mul()(
                inputs, gammas, betas, normalized_shapes, eps
            )
        else:
            Ys = ops.group_layernorm()(inputs, gammas, betas, normalized_shapes, eps)

        for i, Y in enumerate(Ys):
            Y._attrs["is_output"] = True
            Y._attrs["name"] = "output_" + str(i)

        target = detect_target()

        with compile_model(
            Ys,
            target,
            "./tmp",
            testname,
        ) as module:
            B = len(input_shapes)

            logging.info(f"Run test {testname}. Input shapes: {input_shapes}")

            xs_pt = []
            gammas_pt = []
            betas_pt = []
            for shape in input_shapes:
                xs_pt.append(get_random_torch_tensor(shape, dtype=dtype))
                norm_shape = shape[batch_ndim:]
                gamma_pt = (
                    None
                    if gamma_is_none
                    else get_random_torch_tensor(norm_shape, dtype=dtype)
                )
                gammas_pt.append(gamma_pt)
                beta_pt = (
                    None
                    if beta_is_none
                    else get_random_torch_tensor(norm_shape, dtype=dtype)
                )
                betas_pt.append(beta_pt)

            ys_pt = []
            for i in range(B):
                y0 = torch.nn.functional.layer_norm(
                    xs_pt[i],
                    xs_pt[i].size()[batch_ndim:],
                    gammas_pt[i],
                    betas_pt[i],
                    eps=eps,
                )
                if fuse_sigmoid_mul:
                    y = torch.mul(xs_pt[i], torch.sigmoid(y0))
                    ys_pt.append(y)
                else:
                    ys_pt.append(y0)

            num_inputs = len(input_shapes) * (
                1 + (not gamma_is_none) + (not beta_is_none)
            )
            inputs = [0 for i in range(num_inputs)]
            input_name_map = module.get_input_name_to_index_map()
            for i in range(len(input_shapes)):
                inputs[input_name_map[f"X_{i}"]] = xs_pt[i]
                if not gamma_is_none:
                    inputs[input_name_map[f"gamma_{i}"]] = gammas_pt[i]
                if not beta_is_none:
                    inputs[input_name_map[f"beta_{i}"]] = betas_pt[i]
            outputs = [torch.empty_like(y) for y in ys_pt]
            module.run_with_tensors(inputs, outputs)
            # module.benchmark_with_tensors(inputs, outputs)

        for i in range(B):
            logging.debug(f"output: {i}")
            y = outputs[i]
            torch.testing.assert_close(ys_pt[i], y, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            param("float16"),
            param("float32"),
            param("bfloat16"),
        ]
    )
    def test_group_fused_layernorm_sigmoid_mul(self, dtype: str):
        # half4 kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]],
            eps=1e-1,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]],
            use_size_op=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4,
            use_size_op=True,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [
                [1024, 256],
                [1024, 256],
                [1024, 128],
                [1024, 256],
                [1024, 128],
                [1024, 256],
            ],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [
                [2048, 2048],
                [2048, 1024],
            ],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4,
            gamma_is_none=True,
            use_size_op=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4,
            gamma_is_none=True,
            use_size_op=True,
            dtype=dtype,
        )

        # Make sure we test the boundary between being able to fit the arguments in constant memory vs not.
        for num_groups in range(38, 41):
            self._test_group_fused_layernorm_sigmoid_mul(
                [[1024, 256]] * num_groups,
                use_size_op=True,
                dtype=dtype,
            )

        # < 1024 kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16]],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            eps=1e-1,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=True,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            beta_is_none=True,
            use_size_op=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            beta_is_none=True,
            use_size_op=True,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1]],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1], [1, 0], [1, 1]],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128], [1024, 0]],
            dtype=dtype,
        )

        # fallback kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            eps=1e-1,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[128, 1025], [128, 0], [128, 1023]],
            dtype=dtype,
        )
        # Ditto boundary test
        for num_groups_divided_by_3 in range(12, 15):
            self._test_group_fused_layernorm_sigmoid_mul(
                [[1024, 1025], [1024, 1276], [1024, 1023]] * num_groups_divided_by_3,
                dtype=dtype,
            )

        # ND
        self._test_group_fused_layernorm_sigmoid_mul(
            [[2, 512, 256, 16], [2, 512, 128, 4]],
            2,
            use_size_op=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[3, 256, 64], [3, 256, 256], [3, 256, 125]],
            1,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16, 3, 1025], [4, 16, 2, 1276], [4, 16, 1, 1023]],
            2,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16, 1025], [4, 16, 1276], [4, 16, 1023]],
            1,
            gamma_is_none=True,
            beta_is_none=True,
            dtype=dtype,
        )

    @parameterized.expand(
        [
            param("float16"),
            param("float32"),
            param("bfloat16"),
        ]
    )
    def test_group_layernorm(self, dtype: str):
        self._test_group_fused_layernorm_sigmoid_mul(
            [
                [1024, 256],
                [1024, 256],
                [1024, 128],
                [1024, 256],
                [1024, 128],
                [1024, 256],
            ],
            fuse_sigmoid_mul=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=True,
            fuse_sigmoid_mul=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            eps=1e-1,
            fuse_sigmoid_mul=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1], [1, 0], [1, 1]],
            fuse_sigmoid_mul=False,
            dtype=dtype,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[2, 512, 256, 16], [2, 512, 128, 4]],
            2,
            use_size_op=False,
            fuse_sigmoid_mul=False,
            dtype=dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
