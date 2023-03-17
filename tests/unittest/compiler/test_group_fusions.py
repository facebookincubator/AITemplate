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
import logging
import unittest

import torch

from aitemplate import compiler
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    count_ops,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    has_op,
)
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


def _prepare_input_tensors(
    m, nk_groups, dtype, start=0, has_bias=True, only_params=False
):
    inputs = []
    for i, (n, k) in enumerate(nk_groups):
        X = Tensor(
            shape=[m, k],
            dtype=dtype,
            name="x_{}".format(i + start),
            is_input=True,
        )
        W = Tensor(
            shape=[n, k],
            dtype=dtype,
            name="w_{}".format(i + start),
            is_input=True,
        )
        B = Tensor(
            shape=[n],
            dtype=dtype,
            name="b_{}".format(i + start),
            is_input=True,
        )
        if has_bias:
            if only_params:
                inputs.append([W, B])
            else:
                inputs.append([X, W, B])
        else:
            if only_params:
                inputs.append([W])
            else:
                inputs.append([X, W])
    return inputs


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GroupOpTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GroupOpTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_group_layernorm_sigmoid_mul_cat_fusion(
        self,
        input_shapes,
        gamma_is_none=False,
        beta_is_none=False,
        add_size_op=False,
        fuse_sigmoid_mul=True,
        num_group_ops=1,
        should_fail=False,
        dtype="float16",
    ):
        if gamma_is_none or beta_is_none or len(input_shapes) <= 1:
            should_fail = True
        testname = (
            "group_layernorm_sigmoid_mul_fusion"
            if fuse_sigmoid_mul
            else "group_layernorm_fusion"
        )
        _LOGGER.info(
            f"{testname}: input_shapes={input_shapes}, "
            f"gamma_is_none={gamma_is_none}, beta_is_none={beta_is_none}",
        )
        inputs = []
        gammas = []
        betas = []
        normalized_shapes = []
        for i, shape in enumerate(input_shapes):
            inputs.append(
                Tensor(
                    shape=[
                        IntImm(shape[0]),
                        IntImm(shape[1]),
                    ],
                    dtype=dtype,
                    name="X_" + str(i),
                    is_input=True,
                )
            )
            gamma = (
                None
                if gamma_is_none
                else Tensor(
                    shape=[IntImm(shape[1])],
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
                    shape=[IntImm(shape[1])],
                    dtype=dtype,
                    name="beta_" + str(i),
                    is_input=True,
                )
            )
            betas.append(beta)
            if add_size_op:
                size = ops.size()(inputs[-1], 1)
                normalized_shapes.append([size])
            else:
                normalized_shapes.append([IntImm(shape[1])])

        Ys = []

        for i in range(len(input_shapes)):
            Y0 = ops.layernorm()(inputs[i], gammas[i], betas[i], normalized_shapes[i])
            if fuse_sigmoid_mul:
                Y1 = ops.elementwise(FuncEnum.SIGMOID)(Y0)
                Y2 = ops.elementwise(FuncEnum.MUL)(inputs[i], Y1)
                Ys.append(Y2)
            else:
                Ys.append(Y0)

        for i, Y in enumerate(Ys):
            Y._attrs["is_output"] = True
            Y._attrs["name"] = f"output_{i}"

        target = detect_target()
        module = compile_model(
            Ys,
            target,
            "./tmp",
            f"{testname}_{self._test_id}",
        )
        self._test_id += 1

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        group_op = (
            "group_layernorm_sigmoid_mul" if fuse_sigmoid_mul else "group_layernorm"
        )
        if should_fail:
            assert not has_op(sorted_ops, group_op)
            return
        else:
            assert has_op(sorted_ops, group_op)
            assert (
                count_ops(sorted_ops, group_op) == num_group_ops
            ), f"expecting {num_group_ops} {group_op} ops, found {count_ops(sorted_ops, group_op)}"

        B = len(input_shapes)

        _LOGGER.info(
            f"Run test group_layernorm_sigmoid_mul. Input shapes: {input_shapes}",
        )

        xs_pt = []
        gammas_pt = []
        betas_pt = []
        for shape in input_shapes:
            xs_pt.append(get_random_torch_tensor(shape, dtype))
            gamma_pt = (
                None if gamma_is_none else get_random_torch_tensor([shape[1]], dtype)
            )
            gammas_pt.append(gamma_pt)
            beta_pt = (
                None if beta_is_none else get_random_torch_tensor([shape[1]], dtype)
            )
            betas_pt.append(beta_pt)

        ys_pt = []
        for i in range(B):
            y0 = torch.nn.functional.layer_norm(
                xs_pt[i], xs_pt[i].size()[1:], gammas_pt[i], betas_pt[i]
            )
            if fuse_sigmoid_mul:
                y = torch.mul(xs_pt[i], torch.sigmoid(y0))
                ys_pt.append(y)
            else:
                ys_pt.append(y0)

        input_name_to_index = module.get_input_name_to_index_map()
        num_inputs = len(input_shapes) * 3
        inputs = [0 for i in range(num_inputs)]
        for i in range(len(input_shapes)):
            inputs[input_name_to_index[f"X_{i}"]] = xs_pt[i]
            if not gamma_is_none:
                inputs[input_name_to_index[f"gamma_{i}"]] = gammas_pt[i]
            if not beta_is_none:
                inputs[input_name_to_index[f"beta_{i}"]] = betas_pt[i]
        ys = []
        for y_pt in ys_pt:
            ys.append(get_torch_empty_tensor(y_pt.size(), dtype))
        module.run_with_tensors(inputs, ys)
        # module.benchmark_with_tensors(inputs, ys)
        for y_pt, y in zip(ys_pt, ys):
            self.assertTrue(
                torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2),
                f"max diff: {torch.max(y_pt - y)}, min diff: {torch.min(y_pt - y)}",
            )

    def test_group_layernorm_sigmoid_mul_fusion_float16(self):
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 256]], fuse_sigmoid_mul=True
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 256]] * 4, fuse_sigmoid_mul=True
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 128], [128, 256], [128, 125]],
            fuse_sigmoid_mul=True,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[10, 64], [10, 64], [10, 64]],
            beta_is_none=True,
            fuse_sigmoid_mul=True,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 1025], [128, 1276], [128, 1023]],
            gamma_is_none=True,
            fuse_sigmoid_mul=True,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 256]] * 4,
            fuse_sigmoid_mul=False,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[64, 64], [128, 256], [1, 125]],
            fuse_sigmoid_mul=True,
            should_fail=True,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 128], [128, 256], [128, 125]],
            fuse_sigmoid_mul=True,
            add_size_op=True,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 128], [128, 256], [128, 125], [128, 125]],
            fuse_sigmoid_mul=True,
            num_group_ops=2,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 120], [128, 1], [128, 256], [128, 1024]],
            fuse_sigmoid_mul=True,
            num_group_ops=1,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 64]] * 39 + [[128, 256]] * 10,
            fuse_sigmoid_mul=True,
            num_group_ops=2,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 64]] * 50,
            fuse_sigmoid_mul=True,
            num_group_ops=2,
        )

        # ctr_mbl_feed overarch cases
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [
                [2048, 256],
                [2048, 256],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 1024],
            ],
            fuse_sigmoid_mul=True,
            num_group_ops=1,
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[2048, 256], [2048, 256], [2048, 1024]],
            fuse_sigmoid_mul=True,
            num_group_ops=1,
        )

    def test_group_layernorm_sigmoid_mul_fusion_float32(self):
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 256]] * 4, fuse_sigmoid_mul=True, dtype="float32"
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[10, 64], [10, 64], [10, 64]],
            beta_is_none=True,
            fuse_sigmoid_mul=True,
            dtype="float32",
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 256]] * 4,
            fuse_sigmoid_mul=False,
            dtype="float32",
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[64, 64], [128, 256], [1, 125]],
            fuse_sigmoid_mul=True,
            should_fail=True,
            dtype="float32",
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 128], [128, 256], [128, 125]],
            fuse_sigmoid_mul=True,
            add_size_op=True,
            dtype="float32",
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 128], [128, 256], [128, 125], [128, 125]],
            fuse_sigmoid_mul=True,
            num_group_ops=2,
            dtype="float32",
        )
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [[128, 64]] * 39 + [[128, 256]] * 10,
            fuse_sigmoid_mul=True,
            num_group_ops=2,
            dtype="float32",
        )

        # ctr_mbl_feed overarch cases
        self._test_group_layernorm_sigmoid_mul_cat_fusion(
            [
                [2048, 256],
                [2048, 256],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 128],
                [2048, 1024],
            ],
            fuse_sigmoid_mul=True,
            num_group_ops=1,
            dtype="float32",
        )

    def test_layernorm_with_cycles(self):
        """
        The test basically forms the following subgraph:

        layernorm_sigmoid_mul_1 = layernorm_sigmoid_mul(...)
        gemm_rcr_2 = gemm_rcr(layernorm_sigmoid_mul_1)
        layernorm_3 = layernorm(gemm_rcr_2)
        layernorm_4 = layernorm(...)
        gemm_rcr_5 = gemm_rcr(layernorm_4)
        layernorm_sigmoid_mul_6 = layernorm_sigmoid_mul(gemm_rcr_5)

        For example, grouping (layernorm_sigmoid_mul_1, layernorm_sigmoid_mul_6)
        and (gemm_rcr_2, gemm_rcr_5) at the same time would introduce a cycle
        between the fused group ops, because we have the following dependency:
            layernorm_sigmoid_mul_1 -> gemm_rcr_2
            gemm_rcr_5 -> layernrom_sigmoid_mul_6
        """
        torch.manual_seed(0)
        testname = "layernorm_with_cycles_0"
        dtype = "float16"
        batch_sizes = [1, 2048]
        eps = 1e-5

        Input0 = Tensor(
            shape=[IntVar(values=batch_sizes, name="batch"), IntImm(value=1024)],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        reshape_to_shape_0 = [-1, 32, 32]
        reshape_0 = ops.reshape()(Input0, reshape_to_shape_0)

        W0 = Tensor(shape=[IntImm(16), IntImm(32)], name="w0", is_input=True)
        gemm_rcr_0 = ops.gemm_rcr()(reshape_0, W0)

        reshape_to_shape_1 = [-1, 512]
        reshape_1 = ops.reshape()(gemm_rcr_0, reshape_to_shape_1)

        Input1 = Tensor(
            shape=[IntVar(values=batch_sizes, name="batch"), IntImm(value=512)],
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        elementwise_0 = ops.elementwise(func_enum=FuncEnum.MUL)(reshape_1, Input1)

        W1 = Tensor(shape=[IntImm(3821), IntImm(512)], name="w1", is_input=True)
        gemm_rcr_1 = ops.gemm_rcr()(elementwise_0, W1)

        concat_dim = 1
        concatenate_0 = ops.concatenate()([Input1, gemm_rcr_1], concat_dim)

        Gamma0 = Tensor(shape=[IntImm(4333)], name="gamma0", is_input=True)
        Beta0 = Tensor(shape=[IntImm(4333)], name="beta0", is_input=True)
        layernorm_0 = ops.layernorm(normalized_shape=None)(
            concatenate_0, Gamma0, Beta0, [IntImm(4333)], eps
        )

        Input2 = Tensor(
            shape=[IntVar(values=batch_sizes, name="batch"), IntImm(value=256)],
            dtype=dtype,
            name="input2",
            is_input=True,
        )
        W2 = Tensor(shape=[IntImm(256), IntImm(256)], name="w2", is_input=True)
        gemm_rcr_2 = ops.gemm_rcr()(Input2, W2)

        Gamma1 = Tensor(shape=[IntImm(256)], name="gamma1", is_input=True)
        Beta1 = Tensor(shape=[IntImm(256)], name="beta1", is_input=True)
        layernorm_1 = ops.layernorm(normalized_shape=None)(
            gemm_rcr_2, Gamma1, Beta1, [IntImm(256)], eps
        )
        elementwise_1 = ops.elementwise(func_enum=FuncEnum.SIGMOID)(layernorm_1)
        elementwise_2 = ops.elementwise(func_enum=FuncEnum.MUL)(
            gemm_rcr_2, elementwise_1
        )

        W3 = Tensor(shape=[IntImm(2048), IntImm(256)], name="w3", is_input=True)
        gemm_rcr_3 = ops.gemm_rcr()(elementwise_2, W3)

        Gamma2 = Tensor(shape=[IntImm(2048)], name="gamma2", is_input=True)
        Beta2 = Tensor(shape=[IntImm(2048)], name="beta2", is_input=True)
        layernorm_2 = ops.layernorm(normalized_shape=None)(
            gemm_rcr_3, Gamma2, Beta2, [IntImm(2048)], eps
        )

        Input3 = Tensor(
            shape=[IntVar(values=batch_sizes, name="batch"), IntImm(value=1320)],
            dtype=dtype,
            name="input3",
            is_input=True,
        )
        Gamma3 = Tensor(shape=[IntImm(1320)], name="gamma3", is_input=True)
        Beta3 = Tensor(shape=[IntImm(1320)], name="beta3", is_input=True)
        layernorm_3 = ops.layernorm(normalized_shape=None)(
            Input3, Gamma3, Beta3, [IntImm(1320)], eps
        )

        W4 = Tensor(shape=[IntImm(128), IntImm(1320)], name="w4", is_input=True)
        gemm_rcr_4 = ops.gemm_rcr()(layernorm_3, W4)

        Gamma4 = Tensor(shape=[IntImm(128)], name="gamma4", is_input=True)
        Beta4 = Tensor(shape=[IntImm(128)], name="beta4", is_input=True)
        layernorm_4 = ops.layernorm(normalized_shape=None)(
            gemm_rcr_4, Gamma4, Beta4, [IntImm(128)], eps
        )
        elementwise_3 = ops.elementwise(func_enum=FuncEnum.SIGMOID)(layernorm_4)
        elementwise_4 = ops.elementwise(func_enum=FuncEnum.MUL)(
            gemm_rcr_4, elementwise_3
        )

        output_0 = ops.concatenate()(
            [elementwise_4, layernorm_3, layernorm_0, layernorm_2], concat_dim
        )
        output_0._attrs["name"] = "output_0"
        output_0._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            [output_0],
            target,
            "./tmp",
            testname,
        )

        for batch in batch_sizes:
            input0_pt = get_random_torch_tensor([batch, 1024], dtype)
            reshape_0_pt = torch.reshape(input0_pt, reshape_to_shape_0)

            w0_pt = get_random_torch_tensor([16, 32], dtype)
            gemm_rcr_0_pt = torch.nn.functional.linear(reshape_0_pt, w0_pt)

            reshape_1_pt = torch.reshape(gemm_rcr_0_pt, reshape_to_shape_1)

            input1_pt = get_random_torch_tensor([batch, 512], dtype)
            elementwise_0_pt = reshape_1_pt * input1_pt

            w1_pt = get_random_torch_tensor([3821, 512], dtype)
            gemm_rcr_1_pt = torch.nn.functional.linear(elementwise_0_pt, w1_pt)

            concatenate_0_pt = torch.cat([input1_pt, gemm_rcr_1_pt], concat_dim)

            gamma0_pt = get_random_torch_tensor([4333], dtype)
            beta0_pt = get_random_torch_tensor([4333], dtype)
            layernorm_0_pt = torch.nn.functional.layer_norm(
                concatenate_0_pt,
                concatenate_0_pt.size()[1:],
                gamma0_pt,
                beta0_pt,
                eps=eps,
            )

            input2_pt = get_random_torch_tensor([batch, 256], dtype)
            w2_pt = get_random_torch_tensor([256, 256], dtype)
            gemm_rcr_2_pt = torch.nn.functional.linear(input2_pt, w2_pt)

            gamma1_pt = get_random_torch_tensor([256], dtype)
            beta1_pt = get_random_torch_tensor([256], dtype)
            layernorm_1_pt = torch.nn.functional.layer_norm(
                gemm_rcr_2_pt, gemm_rcr_2_pt.size()[1:], gamma1_pt, beta1_pt, eps=eps
            )
            elementwise_1_pt = torch.sigmoid(layernorm_1_pt)
            elementwise_2_pt = torch.mul(gemm_rcr_2_pt, elementwise_1_pt)

            w3_pt = get_random_torch_tensor([2048, 256], dtype)
            gemm_rcr_3_pt = torch.nn.functional.linear(elementwise_2_pt, w3_pt)

            gamma2_pt = get_random_torch_tensor([2048], dtype)
            beta2_pt = get_random_torch_tensor([2048], dtype)
            layernorm_2_pt = torch.nn.functional.layer_norm(
                gemm_rcr_3_pt, gemm_rcr_3_pt.size()[1:], gamma2_pt, beta2_pt, eps=eps
            )

            input3_pt = get_random_torch_tensor([batch, 1320], dtype)
            gamma3_pt = get_random_torch_tensor([1320], dtype)
            beta3_pt = get_random_torch_tensor([1320], dtype)
            layernorm_3_pt = torch.nn.functional.layer_norm(
                input3_pt, input3_pt.size()[1:], gamma3_pt, beta3_pt, eps=eps
            )

            w4_pt = get_random_torch_tensor([128, 1320], dtype)
            gemm_rcr_4_pt = torch.nn.functional.linear(layernorm_3_pt, w4_pt)

            gamma4_pt = get_random_torch_tensor([128], dtype)
            beta4_pt = get_random_torch_tensor([128], dtype)
            layernorm_4_pt = torch.nn.functional.layer_norm(
                gemm_rcr_4_pt, gemm_rcr_4_pt.size()[1:], gamma4_pt, beta4_pt, eps=eps
            )
            elementwise_3_pt = torch.sigmoid(layernorm_4_pt)
            elementwise_4_pt = torch.mul(gemm_rcr_4_pt, elementwise_3_pt)

            output_0_pt = torch.cat(
                [elementwise_4_pt, layernorm_3_pt, layernorm_0_pt, layernorm_2_pt],
                concat_dim,
            )

            inputs = {
                "input0": input0_pt,
                "input1": input1_pt,
                "input2": input2_pt,
                "input3": input3_pt,
                "w0": w0_pt,
                "w1": w1_pt,
                "w2": w2_pt,
                "w3": w3_pt,
                "w4": w4_pt,
                "gamma0": gamma0_pt,
                "beta0": beta0_pt,
                "gamma1": gamma1_pt,
                "beta1": beta1_pt,
                "gamma2": gamma2_pt,
                "beta2": beta2_pt,
                "gamma3": gamma3_pt,
                "beta3": beta3_pt,
                "gamma4": gamma4_pt,
                "beta4": beta4_pt,
            }
            y = torch.empty_like(output_0_pt)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(output_0_pt, y, atol=0.03, rtol=0.03))

    def _test_group_gemm_fusion(
        self,
        m,
        nk_groups,
        has_bias=True,
        has_relu=False,
        has_sigmoid=False,
        should_fail=False,
        dtype="float16",
    ):
        _LOGGER.info(
            f"Running _test_group_gemm_fusion, m = {m}, nk_groups = {nk_groups}, "
            f"has_bias = {has_bias}, has_relu = {has_relu}, has_sigmoid = {has_sigmoid}, "
            f"should_fail = {should_fail}",
        )
        if len(nk_groups) == 1:
            should_fail = True
        op_type = None
        if has_bias:
            if has_relu:
                op = ops.gemm_rcr_bias_relu
                op_type = "group_gemm_rcr_bias_relu"
            elif has_sigmoid:
                op = ops.gemm_rcr_bias_sigmoid
                op_type = "group_gemm_rcr_bias_sigmoid"
            else:
                op = ops.gemm_rcr_bias
                op_type = "group_gemm_rcr_bias"
        else:
            op = ops.gemm_rcr
            op_type = "group_gemm_rcr"

        group_input_tensors = _prepare_input_tensors(
            m, nk_groups, dtype, has_bias=has_bias
        )
        graph = []
        for i, group in enumerate(group_input_tensors):
            Y = op()(*group)
            graph.append(Y)
            Y._attrs["name"] = "y_{}".format(i)
            Y._attrs["is_output"] = True

        target = detect_target()
        with target:
            graph = compiler.transform.toposort(graph)
            compiler.transform.name_graph(graph)
            compiler.transform.mark_param_tensor(graph)
            graph = compiler.transform.fuse_ops(graph)
            graph = compiler.transform.fuse_group_gemm_ops(graph)
            sorted_ops = graph_utils.get_sorted_ops(graph)

            if not should_fail:
                assert has_op(sorted_ops, op_type)
            else:
                assert not has_op(sorted_ops, op_type)

    def test_group_gemm_fusion_float16(self):
        self._test_group_gemm_fusion(1024, [[16, 64], [32, 32]])
        self._test_group_gemm_fusion(1024, [[16, 64], [32, 40]], has_bias=False)
        self._test_group_gemm_fusion(
            1024, [[16, 64], [32, 40], [75, 128]], has_relu=True
        )
        self._test_group_gemm_fusion(
            1024, [[16, 64], [32, 40], [75, 128]], has_sigmoid=True
        )

        # test misalignment
        self._test_group_gemm_fusion(1024, [[16, 44], [32, 32]], should_fail=True)
        self._test_group_gemm_fusion(1024, [[16, 13], [32, 1]], should_fail=True)

    def test_group_gemm_fusion_float32_sm80(self):
        self._test_group_gemm_fusion(32, [[16, 64], [32, 32]], dtype="float32")
        self._test_group_gemm_fusion(
            32, [[16, 64], [32, 40]], has_bias=False, dtype="float32"
        )
        self._test_group_gemm_fusion(
            32, [[16, 64], [32, 40], [75, 128]], has_relu=True, dtype="float32"
        )
        # test misalignment
        self._test_group_gemm_fusion(
            32, [[16, 13], [32, 1]], should_fail=True, dtype="float32"
        )

    def _test_split_group_gemm_fusion(
        self,
        m,
        nk_groups_1,
        nk_groups_2,
        split_dim=1,
        should_fail=False,
        num_group_ops=2,
        dtype="float16",
    ):
        _LOGGER.info(
            f"Running _test_split_group_gemm_fusion, m = {m}, nk_groups_1 = {nk_groups_1}, "
            f"nk_groups_2 = {nk_groups_2}, split_dim = {split_dim}, should_fail: {should_fail}, "
            f"num_group_ops = {num_group_ops}",
        )
        op_type = "group_gemm_rcr_bias"

        inputs1 = _prepare_input_tensors(
            m, nk_groups_1, dtype, has_bias=True, only_params=True
        )
        inputs2 = _prepare_input_tensors(
            m,
            nk_groups_2,
            dtype,
            start=len(nk_groups_1),
            has_bias=True,
            only_params=False,
        )

        if split_dim == 1:
            split_sizes = [k for n, k in nk_groups_1]
            K = sum(split_sizes)
            X = Tensor(
                shape=[m, K],
                dtype=dtype,
                name="input",
                is_input=True,
            )
        else:
            split_sizes = m
            X = Tensor(
                shape=[m * len(nk_groups_1), nk_groups_1[0][1]],
                dtype=dtype,
                name="input",
                is_input=True,
            )

        Y1s = ops.split()(X, split_sizes, split_dim)

        graph = []
        for i, inputs in enumerate(inputs1):
            inputs = [Y1s[i]] + inputs
            Y = ops.gemm_rcr_bias()(*inputs)
            graph.append(Y)
            Y._attrs["name"] = "y_{}".format(i)
            Y._attrs["is_output"] = True

        for i, inputs in enumerate(inputs2):
            Y = ops.gemm_rcr_bias()(*inputs)
            graph.append(Y)
            Y._attrs["name"] = "y_{}".format(len(nk_groups_1) + i)
            Y._attrs["is_output"] = True

        target = detect_target()
        with target:
            graph = compiler.transform.toposort(graph)
            compiler.transform.name_graph(graph)
            compiler.transform.mark_param_tensor(graph)
            graph = compiler.transform.fuse_ops(graph)
            graph = compiler.transform.fuse_group_gemm_ops(graph)
            graph = compiler.transform.transform_strided_ops(graph)
            sorted_ops = graph_utils.get_sorted_ops(graph)

            if should_fail:
                assert has_op(sorted_ops, "split")
                assert count_ops(sorted_ops, op_type) == num_group_ops
            else:
                assert not has_op(sorted_ops, "split")
                assert count_ops(sorted_ops, op_type) == num_group_ops

    def test_split_group_gemm_fusion_float16(self):
        self._test_split_group_gemm_fusion(
            1024, [[16, 64], [16, 40], [16, 128]], [[1, 16], [3, 48]], num_group_ops=2
        )
        self._test_split_group_gemm_fusion(
            48,
            [[16, 64], [16, 64], [16, 64]],
            [[1, 16], [3, 48]],
            split_dim=0,
            should_fail=True,
            num_group_ops=1,
        )
        self._test_split_group_gemm_fusion(
            48,
            [[16, 63], [16, 64], [16, 64]],
            [[1, 16], [3, 48]],
            should_fail=True,
            num_group_ops=1,
        )

    def test_split_group_gemm_fusion_float32_sm80(self):
        self._test_split_group_gemm_fusion(
            32,
            [[16, 64], [16, 40], [16, 128]],
            [[1, 16], [3, 48]],
            num_group_ops=2,
            dtype="float32",
        )
        self._test_split_group_gemm_fusion(
            48,
            [[16, 64], [16, 64], [16, 64]],
            [[1, 16], [3, 48]],
            split_dim=0,
            should_fail=True,
            num_group_ops=1,
            dtype="float32",
        )


filter_test_cases_by_test_env(GroupOpTestCase)

if __name__ == "__main__":
    unittest.main()
