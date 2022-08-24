# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for FusedLayernormSigmoidMul Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FusedLayernormSigmoidMulTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FusedLayernormSigmoidMulTestCase, self).__init__(*args, **kwargs)
        torch.manual_seed(0)
        self._atol = 1e-2
        self._rtol = 1e-3

    def _test_fused_layernorm_sigmoid_mul(
        self,
        MS=(),
        NS=(16,),
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
    ):
        logging.info(
            f"_test_fused_layernorm_sigmoid_mul: M={MS}, N={NS}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}"
        )
        assert isinstance(MS, (list, tuple))
        assert isinstance(NS, (list, tuple))

        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=[1, 1024]), *MS, *NS],
            dtype="float16",
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=NS,
                dtype="float16",
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=NS,
                dtype="float16",
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
        with gen_execution_module(
            X6, target, "./tmp", "fused_layernorm_sigmoid_mul_test"
        ) as module:
            for batch_size in [50, 900, 1024]:
                logging.info(
                    f"Run test layernorm_sigmoid_mul. Problem size {[batch_size,] + list(MS) + list(NS)}"
                )
                x1_pt = torch.randn(batch_size, *MS, *NS).cuda().half()
                if gamma_is_none:
                    x2_pt = None
                else:
                    x2_pt = torch.randn(NS).cuda().half()
                if beta_is_none:
                    x3_pt = None
                else:
                    x3_pt = torch.randn(NS).cuda().half()

                x4_pt = torch.nn.functional.layer_norm(x1_pt, NS, x2_pt, x3_pt, eps=eps)
                x6_pt = torch.mul(x1_pt, torch.sigmoid(x4_pt))

                inputs = {"X": x1_pt}
                if not gamma_is_none:
                    inputs["gamma"] = x2_pt
                if not beta_is_none:
                    inputs["beta"] = x3_pt
                x6 = torch.empty([batch_size, *MS, *NS]).cuda().half()
                module.RunWithTensors(inputs, [x6])
                self.assertTrue(
                    torch.allclose(x6, x6_pt, atol=self._atol, rtol=self._rtol),
                    f"max diff: {torch.max(x6 - x6_pt) if x6_pt.numel() > 0 else 0}, "
                    f"min diff: {torch.min(x6 - x6_pt) if x6_pt.numel() > 0 else 0}",
                )

    def test_fused_layernorm_sigmoid_mul(self):
        for eps in (1e-5, 1e-1):
            # half4 kernel
            self._test_fused_layernorm_sigmoid_mul(NS=(1496,), eps=eps)
            # block_size = n kernel
            self._test_fused_layernorm_sigmoid_mul(NS=(515,), eps=eps)
            # block_size = 512 kernel
            self._test_fused_layernorm_sigmoid_mul(NS=(1055,), eps=eps)

        # test ND inputs
        eps = 1e-5
        # half4 kernel
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 2), NS=(64, 8), eps=eps)
        # block_size = n kernel
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 2), NS=(213, 2), eps=eps)
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 2), NS=(3, 2), eps=eps)
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 2), NS=(1, 1), eps=eps)
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 2), NS=(0, 1), eps=eps)
        # block_size = 512 kernel
        self._test_fused_layernorm_sigmoid_mul(MS=(2, 4), NS=(1055, 5), eps=eps)

        self._test_fused_layernorm_sigmoid_mul(
            NS=(1496,), gamma_is_none=True, beta_is_none=True
        )
        self._test_fused_layernorm_sigmoid_mul(
            NS=(515,), gamma_is_none=True, beta_is_none=True
        )
        for use_size_op in (True, False):
            self._test_fused_layernorm_sigmoid_mul(NS=(1055,), use_size_op=use_size_op)
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1055,),
                gamma_is_none=True,
                beta_is_none=True,
                use_size_op=use_size_op,
            )
            self._test_fused_layernorm_sigmoid_mul(
                NS=(1496,), gamma_is_none=True, use_size_op=use_size_op
            )
            self._test_fused_layernorm_sigmoid_mul(
                NS=(515,), beta_is_none=True, use_size_op=use_size_op
            )

    # dim0 is batch size
    def _test_batch_fused_layernorm_sigmoid_mul(
        self, M, N, gamma_is_none=False, beta_is_none=False, use_size_op=False, eps=1e-5
    ):
        logging.info(
            f"_test_batch_fused_layernorm_sigmoid_mul: M={M}, N={N}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}"
        )
        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(M), IntImm(N)],
            dtype="float16",
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(N)],
                dtype="float16",
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=[IntVar(name="input_batch", values=[2, 32]), IntImm(N)],
                dtype="float16",
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
        with gen_execution_module(
            X4, target, "./tmp", f"batch_fused_layernorm_sigmoid_mul_{M}_{N}_test"
        ) as module:
            for batch_size in [2, 16, 32]:
                logging.info(
                    "Run test batch_layernorm_sigmoid_mul. Problem size [{}, {}, {}]".format(
                        batch_size, M, N
                    )
                )
                xs_pt = [torch.randn(M, N).cuda().half() for i in range(batch_size)]
                if gamma_is_none:
                    gammas_pt = [None] * batch_size
                else:
                    gammas_pt = [
                        torch.randn(N).cuda().half() for i in range(batch_size)
                    ]
                if beta_is_none:
                    betas_pt = [None] * batch_size
                else:
                    betas_pt = [torch.randn(N).cuda().half() for i in range(batch_size)]

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
                x4 = torch.empty([batch_size, M, N]).cuda().half()
                module.RunWithTensors(inputs, [x4])
                self.assertTrue(
                    torch.allclose(x4, y_t, atol=self._atol, rtol=self._rtol),
                    f"max diff: {torch.max(x4 - y_t) if y_t.numel() > 0 else 0}, "
                    f"min diff: {torch.min(x4 - y_t) if y_t.numel() > 0 else 0}",
                )

    # dim1 is the batch size
    def _test_batch_fused_layernorm_sigmoid_mul_dim1(
        self, B, N, gamma_is_none=False, beta_is_none=False
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
            dtype="float16",
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=[IntImm(B), IntImm(N)],
                dtype="float16",
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=[IntImm(B), IntImm(N)],
                dtype="float16",
                name="beta",
                is_input=True,
            )
        X4 = ops.batch_layernorm_sigmoid_mul(normalized_shape=[IntImm(N)])(X1, X2, X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        with gen_execution_module(
            X4,
            target,
            "./tmp",
            f"batch_fused_layernorm_sigmoid_mul_dim1_{B}_{N}_test",
        ) as module:
            for M in [128, 1024]:
                logging.info(
                    "Run test batch_layernorm_sigmoid_mul. Problem size [{}, {}, {}]".format(
                        B, M, N
                    )
                )
                xs_pt = [torch.randn(M, N).cuda().half() for i in range(B)]
                if gamma_is_none:
                    gammas_pt = [None] * B
                else:
                    gammas_pt = [torch.randn(N).cuda().half() for i in range(B)]
                if beta_is_none:
                    betas_pt = [None] * B
                else:
                    betas_pt = [torch.randn(N).cuda().half() for i in range(B)]

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
                x4 = torch.empty([B, M, N]).cuda().half()
                module.RunWithTensors(inputs, [x4])
                self.assertTrue(
                    torch.allclose(x4, y_t, atol=self._atol, rtol=self._rtol),
                    f"max diff: {torch.max(x4 - y_t) if y_t.numel() > 0 else 0}, "
                    f"min diff: {torch.min(x4 - y_t) if y_t.numel() > 0 else 0}",
                )

    def test_batch_fused_layernorm_sigmoid_mul(self):
        for eps in (1e-5, 1e-1):
            self._test_batch_fused_layernorm_sigmoid_mul(512, 1024, eps=eps)
            self._test_batch_fused_layernorm_sigmoid_mul(512, 64, eps=eps)

        self._test_batch_fused_layernorm_sigmoid_mul(
            512, 1024, gamma_is_none=True, beta_is_none=True
        )
        self._test_batch_fused_layernorm_sigmoid_mul(
            512, 64, gamma_is_none=True, beta_is_none=True
        )
        for use_size_op in (True, False):
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024, 1055, use_size_op=use_size_op, eps=1e-1
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024, 1055, use_size_op=use_size_op
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                1024,
                1055,
                gamma_is_none=True,
                beta_is_none=True,
                use_size_op=use_size_op,
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                512, 1024, gamma_is_none=True, use_size_op=use_size_op
            )
            self._test_batch_fused_layernorm_sigmoid_mul(
                512, 1024, beta_is_none=True, use_size_op=use_size_op
            )

        self._test_batch_fused_layernorm_sigmoid_mul_dim1(1, 512)
        self._test_batch_fused_layernorm_sigmoid_mul_dim1(16, 512)

        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            1, 512, gamma_is_none=True, beta_is_none=True
        )
        self._test_batch_fused_layernorm_sigmoid_mul_dim1(
            16, 512, gamma_is_none=True, beta_is_none=True
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
    ):
        testname = (
            "group_fused_layernorm_sigmoid_mul_test"
            if fuse_sigmoid_mul
            else "group_layernorm_test"
        )
        logging.info(
            f"{testname}: input_shapes={input_shapes}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}, "
            f"use_size_op={use_size_op}"
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
                    dtype="float16",
                    name="X_" + str(i),
                    is_input=True,
                )
            )
            gamma = (
                None
                if gamma_is_none
                else Tensor(
                    shape=[IntImm(n) for n in shape[batch_ndim:]],
                    dtype="float16",
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
                    dtype="float16",
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

        with gen_execution_module(
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
                xs_pt.append(torch.randn(shape).cuda().half())
                norm_shape = shape[batch_ndim:]
                gamma_pt = (
                    None if gamma_is_none else torch.randn(norm_shape).cuda().half()
                )
                gammas_pt.append(gamma_pt)
                beta_pt = (
                    None if beta_is_none else torch.randn(norm_shape).cuda().half()
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
            input_name_map = module.GetInputNameToIndexMap()
            for i in range(len(input_shapes)):
                inputs[input_name_map[f"X_{i}"]] = xs_pt[i]
                if not gamma_is_none:
                    inputs[input_name_map[f"gamma_{i}"]] = gammas_pt[i]
                if not beta_is_none:
                    inputs[input_name_map[f"beta_{i}"]] = betas_pt[i]
            outputs = [torch.empty_like(y) for y in ys_pt]
            module.RunWithTensors(inputs, outputs)
            # module.BenchmarkWithTensors(inputs, outputs)

        for i in range(B):
            logging.debug("output: {}".format(str(i)))
            y = outputs[i]
            self.assertTrue(
                torch.allclose(ys_pt[i], y, atol=self._atol, rtol=self._rtol),
                f"max diff: {torch.max(ys_pt[i]- y) if y.numel() > 0 else 0}, "
                f"min diff: {torch.min(ys_pt[i] - y) if y.numel() > 0 else 0}",
            )

    def test_group_fused_layernorm_sigmoid_mul(self):
        # half4 kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]], eps=1e-1
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]], use_size_op=False
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4, use_size_op=True
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [
                [1024, 256],
                [1024, 256],
                [1024, 128],
                [1024, 256],
                [1024, 128],
                [1024, 256],
            ]
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [
                [2048, 2048],
                [2048, 1024],
            ]
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=False,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4, gamma_is_none=True, use_size_op=False
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256]] * 4, gamma_is_none=True, use_size_op=True
        )

        # Make sure we test the boundary between being able to fit the arguments in constant memory vs not.
        for num_groups in range(55, 61):
            self._test_group_fused_layernorm_sigmoid_mul(
                [[1024, 256]] * num_groups, use_size_op=True
            )

        # < 1024 kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16]],
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]], eps=1e-1
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]]
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=True,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            beta_is_none=True,
            use_size_op=False,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            beta_is_none=True,
            use_size_op=True,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1]],
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1], [1, 0], [1, 1]],
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 256], [1024, 128], [1024, 0]]
        )

        # fallback kernel
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]], eps=1e-1
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]]
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            gamma_is_none=True,
            beta_is_none=True,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[128, 1025], [128, 0], [128, 1023]]
        )
        # Ditto boundary test
        for num_groups_divided_by_3 in range(18, 21):
            self._test_group_fused_layernorm_sigmoid_mul(
                [[1024, 1025], [1024, 1276], [1024, 1023]] * num_groups_divided_by_3
            )

        # ND
        self._test_group_fused_layernorm_sigmoid_mul(
            [[2, 512, 256, 16], [2, 512, 128, 4]], 2, use_size_op=False
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[3, 256, 64], [3, 256, 256], [3, 256, 125]], 1
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16, 3, 1025], [4, 16, 2, 1276], [4, 16, 1, 1023]],
            2,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[4, 16, 1025], [4, 16, 1276], [4, 16, 1023]],
            1,
            gamma_is_none=True,
            beta_is_none=True,
        )

    def test_group_layernorm(self):
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
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 64], [1024, 256], [1024, 125]],
            gamma_is_none=True,
            beta_is_none=True,
            use_size_op=True,
            fuse_sigmoid_mul=False,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1024, 1025], [1024, 1276], [1024, 1023]],
            eps=1e-1,
            fuse_sigmoid_mul=False,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[1, 1], [1, 0], [1, 1]],
            fuse_sigmoid_mul=False,
        )
        self._test_group_fused_layernorm_sigmoid_mul(
            [[2, 512, 256, 16], [2, 512, 128, 4]],
            2,
            use_size_op=False,
            fuse_sigmoid_mul=False,
        )


if __name__ == "__main__":
    unittest.main()
