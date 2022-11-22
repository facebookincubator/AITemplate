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

from aitemplate import compiler
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import count_ops, has_op
from aitemplate.utils import graph_utils, logger


def _prepare_input_tensors(m, nk_groups, start=0, has_bias=True, only_params=False):
    inputs = []
    for i, (n, k) in enumerate(nk_groups):
        X = Tensor(
            shape=[m, k],
            dtype="float16",
            name="x_{}".format(i + start),
            is_input=True,
        )
        W = Tensor(
            shape=[n, k],
            dtype="float16",
            name="w_{}".format(i + start),
            is_input=True,
        )
        B = Tensor(
            shape=[n],
            dtype="float16",
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
    ):
        if gamma_is_none or beta_is_none or len(input_shapes) <= 1:
            should_fail = True
        testname = (
            "group_layernorm_sigmoid_mul_fusion"
            if fuse_sigmoid_mul
            else "group_layernorm_fusion"
        )
        logger.info(
            __file__,
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
                    dtype="float16",
                    name="X_" + str(i),
                    is_input=True,
                )
            )
            gamma = (
                None
                if gamma_is_none
                else Tensor(
                    shape=[IntImm(shape[1])],
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
                    shape=[IntImm(shape[1])],
                    dtype="float16",
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

        logger.info(
            __file__,
            f"Run test group_layernorm_sigmoid_mul. Input shapes: {input_shapes}",
        )

        xs_pt = []
        gammas_pt = []
        betas_pt = []
        for shape in input_shapes:
            xs_pt.append(torch.randn(shape).cuda().half())
            gamma_pt = None if gamma_is_none else torch.randn(shape[1]).cuda().half()
            gammas_pt.append(gamma_pt)
            beta_pt = None if beta_is_none else torch.randn(shape[1]).cuda().half()
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
            ys.append(torch.empty(y_pt.size()).cuda().half())
        module.run_with_tensors(inputs, ys)
        # module.benchmark_with_tensors(inputs, ys)
        for y_pt, y in zip(ys_pt, ys):
            self.assertTrue(
                torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2),
                f"max diff: {torch.max(y_pt - y)}, min diff: {torch.min(y_pt - y)}",
            )

    def test_group_layernorm_sigmoid_mul_fusion(self):
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

    def _test_group_gemm_fusion(
        self,
        m,
        nk_groups,
        has_bias=True,
        has_relu=False,
        has_sigmoid=False,
        should_fail=False,
    ):
        logger.info(
            __file__,
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

        group_input_tensors = _prepare_input_tensors(m, nk_groups, has_bias=has_bias)
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

    def test_group_gemm_fusion(self):
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

    def _test_split_group_gemm_fusion(
        self,
        m,
        nk_groups_1,
        nk_groups_2,
        split_dim=1,
        should_fail=False,
        num_group_ops=2,
    ):
        logger.info(
            __file__,
            f"Running _test_split_group_gemm_fusion, m = {m}, nk_groups_1 = {nk_groups_1}, "
            f"nk_groups_2 = {nk_groups_2}, split_dim = {split_dim}, should_fail: {should_fail}, "
            f"num_group_ops = {num_group_ops}",
        )
        op_type = "group_gemm_rcr_bias"

        inputs1 = _prepare_input_tensors(
            m, nk_groups_1, has_bias=True, only_params=True
        )
        inputs2 = _prepare_input_tensors(
            m, nk_groups_2, start=len(nk_groups_1), has_bias=True, only_params=False
        )

        if split_dim == 1:
            split_sizes = [k for n, k in nk_groups_1]
            K = sum(split_sizes)
            X = Tensor(
                shape=[m, K],
                dtype="float16",
                name="input",
                is_input=True,
            )
        else:
            split_sizes = m
            X = Tensor(
                shape=[m * len(nk_groups_1), nk_groups_1[0][1]],
                dtype="float16",
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

    def test_split_group_gemm_fusion(self):
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


if __name__ == "__main__":
    unittest.main()
