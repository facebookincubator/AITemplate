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

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target


_LOGGER = logging.getLogger(__name__)


def _prepare_input_tensors(m, nk_groups, start=0, has_bias=True):
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
            inputs.append([X, W, B])
        else:
            inputs.append([X, W])
    return inputs


def _prepare_batch_input_tensors(b, m, n, k, has_bias=False):
    inputs = []
    X = Tensor(
        shape=[b, m, k],
        dtype="float16",
        name="x",
        is_input=True,
    )
    W = Tensor(
        shape=[b, n, k],
        dtype="float16",
        name="w",
        is_input=True,
    )
    B = Tensor(
        shape=[n],
        dtype="float16",
        name="b",
        is_input=True,
    )
    if has_bias:
        inputs = [X, W, B]
    else:
        inputs = [X, W]
    return inputs


def _prepare_inputs(m, nk_groups, repeats=10, has_bias=True):
    inputs = []
    for _ in range(repeats):
        inputs.append([])
        for n, k in nk_groups:
            x_pt = torch.randn(m, k).half().cuda()
            w_pt = torch.randn(n, k).half().cuda()
            b_pt = torch.randn(n).half().cuda()
            if has_bias:
                inputs[-1].extend((x_pt, w_pt, b_pt))
            else:
                inputs[-1].extend((x_pt, w_pt))
    return inputs


def _prepare_batch_inputs(non_batch_inputs, has_bias=False):
    inputs = [[] for i in range(len(non_batch_inputs))]
    for i, non_batch_input in enumerate(non_batch_inputs):
        n = 3 if has_bias else 2
        to_be_stacked = [[] for j in range(n)]
        for j, inp in enumerate(non_batch_input):
            to_be_stacked[j % n].append(inp)
        for j in range(n):
            inputs[i].append(torch.stack(to_be_stacked[j], dim=0))
    return inputs


def _prepare_outputs(output_tensors):
    def _to_int_list(shape):
        result = []
        for d in shape:
            assert isinstance(d, IntImm)
            result.append(d._attrs["values"][0])
        return result

    output_shapes = [_to_int_list(output._attrs["shape"]) for output in output_tensors]
    outputs = [torch.empty(shape).half().cuda() for shape in output_shapes]
    return outputs


def _prepare_group_gemm_ait_module(
    m, nk_groups_1, nk_groups_2=None, test_idx=0, has_bias=True
):
    output_tensors = []

    OP = ops.group_gemm_rcr_bias() if has_bias else ops.group_gemm_rcr()
    Ys = OP(operand_groups=_prepare_input_tensors(m, nk_groups_1, has_bias=has_bias))
    output_tensors.extend(Ys)

    if nk_groups_2:
        OP_2 = ops.group_gemm_rcr_bias()
        Ys_2 = OP_2(
            operand_groups=_prepare_input_tensors(
                m, nk_groups_2, len(nk_groups_1), has_bias=has_bias
            )
        )
        output_tensors.extend(Ys_2)
    else:
        nk_groups_2 = []

    for i, Y in enumerate(output_tensors):
        Y._attrs["name"] = "y_{}".format(i)
        Y._attrs["is_output"] = True

    target = detect_target()
    module = compile_model(
        output_tensors,
        target,
        "./tmp",
        f"group_gemm_rcr_{'bias_' if has_bias else ''}{m}_{len(nk_groups_1)}_{len(nk_groups_2)}_{test_idx}",
    )
    outputs = _prepare_outputs(output_tensors)
    return outputs, module


def _prepare_gemm_ait_module(m, nk_groups, test_idx=0, has_bias=True):
    group_input_tensors = _prepare_input_tensors(m, nk_groups, has_bias=has_bias)
    input_tensors = []
    output_tensors = []
    for group in group_input_tensors:
        input_tensors.extend(group)
        Y = ops.gemm_rcr_bias()(*group) if has_bias else ops.gemm_rcr()(*group)
        output_tensors.append(Y)

    for i, Y in enumerate(output_tensors):
        Y._attrs["name"] = "y_{}".format(i)
        Y._attrs["is_output"] = True

    target = detect_target()
    module = compile_model(
        output_tensors,
        target,
        "./tmp",
        f"gemm_rcr_{'bias_' if has_bias else ''}{m}_{len(nk_groups)}_{test_idx}",
    )
    outputs = _prepare_outputs(output_tensors)
    return outputs, module


def _prepare_bmm_ait_module(b, m, n, k, has_bias=False):
    assert (
        not has_bias
    ), "bmm_rcr_bias is not implemented! has_bias has to be false for now"
    input_tensors = _prepare_batch_input_tensors(b, m, n, k, has_bias=has_bias)
    Y = ops.bmm_rcr()(*input_tensors)
    Y._attrs["name"] = "y"
    Y._attrs["is_output"] = True

    target = detect_target()
    module = compile_model(
        [Y],
        target,
        "./tmp",
        f"bmm_rcr_{'bias_' if has_bias else ''}{b}_{m}_{n}_{k}",
    )
    outputs = _prepare_outputs([Y])
    return outputs, module


def _benchmark(count, inputs_repeats, warmup, inputs, outputs, module, test_name):
    for i in range(warmup):
        module.run_with_tensors(inputs[i % inputs_repeats], outputs, sync=False)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(count):
        module.run_with_tensors(inputs[i % inputs_repeats], outputs, sync=False)
    end_event.record()
    torch.cuda.synchronize()
    _LOGGER.warning(
        f"{test_name} benchmark, duration: {start_event.elapsed_time(end_event) / count}ms",
    )


@unittest.skipIf(detect_target().in_ci_env(), "benchmark")
class GroupGemmBenchTestCase(unittest.TestCase):
    def test_rcr(self):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return
        X1 = Tensor(shape=[M, K1], dtype="float16", name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype="float16", name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype="float16", name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype="float16", name="w2", is_input=True)
        B1 = Tensor(shape=[N1], dtype="float16", name="b1", is_input=True)
        B2 = Tensor(shape=[N2], dtype="float16", name="b2", is_input=True)
        OP = ops.group_gemm_rcr_bias()
        Y1, Y2 = OP(operand_groups=[[X1, W1, B1], [X2, W2, B2]])
        Y1._attrs["name"] = "y1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "y2"
        Y2._attrs["is_output"] = True
        module = compile_model([Y1, Y2], target, "./tmp", "group_gemm_rcr_bias")
        X1_pt = torch.randn(M, K1).cuda().half()
        X2_pt = torch.randn(M, K2).cuda().half()
        W1_pt = torch.randn(N1, K1).cuda().half()
        W2_pt = torch.randn(N2, K2).cuda().half()
        B1_pt = torch.randn(N1).cuda().half()
        B2_pt = torch.randn(N2).cuda().half()
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt, bias=B1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt, bias=B2_pt)

        inputs = [
            X1_pt,
            W1_pt,
            B1_pt,
            X2_pt,
            W2_pt,
            B2_pt,
        ]
        y1 = torch.empty([M, N1]).cuda().half()
        y2 = torch.empty([M, N2]).cuda().half()
        module.run_with_tensors(inputs, [y1, y2])
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(Y2_pt, y2, atol=1e-1, rtol=1e-1))

    def _benchmark_rcr(
        self,
        m,
        nk_groups_1,
        nk_groups_2=None,
        test_idx=0,
        test_name="",
        benchmark_non_group=False,
    ):
        _LOGGER.warning(
            f"{test_name} benchmark, m: {m}, nk groups: {nk_groups_1}, {nk_groups_2}",
        )
        WARMUP = 10000
        COUNT = 50000
        nk_groups = nk_groups_1 + nk_groups_2
        total_size = 0
        for n, k in nk_groups:
            total_size = (m * k + n * k) * 2  # half_t is 2 bytes.
        inputs_repeats = int(
            (40 * 1024 * 1024) / total_size * 2
        )  # Makes sure input size is larger than L2 cache.

        if nk_groups_2 is None:
            nk_groups_2 = []
        inputs = _prepare_inputs(m, nk_groups, inputs_repeats)

        group_gemm_outputs, group_gemm_module = _prepare_group_gemm_ait_module(
            m, nk_groups_1, nk_groups_2, test_idx
        )
        _benchmark(
            COUNT,
            inputs_repeats,
            WARMUP,
            inputs,
            group_gemm_outputs,
            group_gemm_module,
            f"{test_name}: group_gemm_batch_size_{m}",
        )

        if benchmark_non_group:
            nk_groups = nk_groups_1 + nk_groups_2
            gemm_outputs, gemm_module = _prepare_gemm_ait_module(m, nk_groups, test_idx)
            _benchmark(
                COUNT,
                inputs_repeats,
                WARMUP,
                inputs,
                gemm_outputs,
                gemm_module,
                f"{test_name}: gemm_batch_size_{m}",
            )

    def _benchmark_batch_rcr(self, b, m, n, k, test_name=""):
        _LOGGER.warning(
            f"{test_name} benchmark, b: {b}, m: {m}, n: {n}, k: {k}",
        )
        WARMUP = 10000
        COUNT = 50000
        nk_groups = [[n, k]] * b
        total_size = (m * k + n * k) * b * 2  # half_t is 2 bytes.
        inputs_repeats = int(
            (40 * 1024 * 1024) / total_size * 2
        )  # Makes sure input size is larger than L2 cache.

        group_gemm_inputs = _prepare_inputs(
            m, nk_groups, inputs_repeats, has_bias=False
        )
        group_gemm_outputs, group_gemm_module = _prepare_group_gemm_ait_module(
            m, nk_groups, has_bias=False
        )
        _benchmark(
            COUNT,
            inputs_repeats,
            WARMUP,
            group_gemm_inputs,
            group_gemm_outputs,
            group_gemm_module,
            f"{test_name}: batch_{b}_group_gemm_{m}_{n}_{k}",
        )
        gemm_outputs, gemm_module = _prepare_gemm_ait_module(
            m, nk_groups, has_bias=False
        )
        _benchmark(
            COUNT,
            inputs_repeats,
            WARMUP,
            group_gemm_inputs,
            gemm_outputs,
            gemm_module,
            f"{test_name}: batch_{b}_normal_gemm_{m}_{n}_{k}",
        )
        gemm_output_cat = torch.stack(gemm_outputs, dim=0)
        bmm_inputs = _prepare_batch_inputs(group_gemm_inputs, has_bias=False)
        bmm_outputs, bmm_module = _prepare_bmm_ait_module(b, m, n, k, has_bias=False)
        _benchmark(
            COUNT,
            inputs_repeats,
            WARMUP,
            bmm_inputs,
            bmm_outputs,
            bmm_module,
            f"batch_{b}_bmm_{m}_{n}_{k}",
        )
        self.assertTrue(
            torch.allclose(gemm_output_cat, bmm_outputs[0], atol=1e-1, rtol=1e-1)
        )

    def test_rcr_benchmark_1(self):
        group1 = [
            [512, 704],
            [256, 704],
            [512, 120],
            [256, 120],
            [256, 328],
            [256, 328],
            [128, 480],
            [256, 480],
        ]
        group2 = [[256, 3200], [128, 3200]]
        for B in (1024, 2048):
            # two separate kernels
            self._benchmark_rcr(
                B,
                group1,
                group2,
            )

    def test_rcr_benchmark_2(self):
        group1 = [
            [512, 704],
            [256, 704],
            [512, 120],
            [256, 120],
            [256, 328],
            [256, 328],
            [128, 480],
            [256, 480],
        ]
        group2 = [[256, 3200], [128, 3200]]
        groups = group1 + group2
        for B in (1024, 2048):
            # out of order
            self._benchmark_rcr(
                B,
                groups,
                [],
                0,
            )

    def test_rcr_benchmark_3(self):
        group1 = [
            [512, 704],
            [256, 704],
            [512, 120],
            [256, 120],
            [256, 328],
            [256, 328],
            [128, 480],
            [256, 480],
        ]
        group2 = [[256, 3200], [128, 3200]]
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                group2 + group1,
                [],
                1,
            )

    def test_rcr_benchmark_4(self):
        group1 = [
            [512, 704],
            [256, 704],
            [512, 120],
            [256, 120],
            [256, 328],
            [256, 328],
            [128, 480],
            [256, 480],
        ]
        group2 = [[256, 3200], [128, 3200]]
        groups = group1 + group2
        for B in (1024, 2048):
            # order by decreasing k
            groups.sort(key=lambda i: i[1], reverse=True)
            self._benchmark_rcr(
                B,
                groups,
                [],
                2,
            )

    def test_rcr_benchmark_5(self):
        group1 = [
            [512, 704],
            [256, 704],
            [512, 120],
            [256, 120],
            [256, 328],
            [256, 328],
            [128, 480],
            [256, 480],
        ]
        group2 = [[256, 3200], [128, 3200]]
        groups = group1 + group2
        for B in (1024, 2048):
            # order by increasing k
            groups.sort(key=lambda i: i[1], reverse=False)
            self._benchmark_rcr(
                B,
                groups,
                [],
                3,
            )

    def test_ads_20x_inline_10_groups(self):
        groups = [
            [200, 72],
            [200, 64],
            [200, 120],
            [200, 120],
            [200, 64],
            [200, 72],
            [200, 64],
            [600, 2048],
            [200, 120],
            [200, 120],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="20_inline_cvr_10_groups",
            )

    def test_ads_20x_inline_6_groups(self):
        groups = [
            [1056, 144],
            [528, 2400],
            [528, 360],
            [656, 176],
            [656, 112],
            [200, 64],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="20_inline_cvr_6_groups",
            )

    def test_ads_20x_inline_2_groups_1(self):
        groups = [
            [256, 256],
            [256, 256],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="20_inline_cvr_2_groups_1",
            )

    def test_ads_20x_inline_2_groups_2(self):
        groups = [
            [22560, 256],
            [6000, 256],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="20_inline_cvr_2_groups_2",
            )

    def test_ads_dhen_inline_15_groups(self):
        groups = [
            [768, 144],
            [384, 2400],
            [384, 360],
            [512, 176],
            [512, 112],
            [128, 64],
            [128, 72],
            [128, 64],
            [128, 64],
            [128, 72],
            [128, 120],
            [128, 120],
            [128, 64],
            [128, 120],
            [128, 120],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="dhen_inline_cvr_15_groups",
            )

    def test_ads_dhen_inline_2_groups_1(self):
        groups = [
            [256, 4096],
            [256, 4096],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="dhen_inline_cvr_2_groups_1",
            )

    def test_ads_dhen_inline_2_groups_2(self):
        groups = [
            [47136, 256],
            [4096, 256],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="dhen_inline_cvr_2_groups_2",
            )

    def test_ads_17x_inline_11_groups(self):
        groups = [
            [128, 64],
            [128, 120],
            [128, 72],
            [128, 72],
            [128, 120],
            [128, 120],
            [128, 64],
            [128, 64],
            [128, 120],
            [128, 64],
            [384, 2048],
        ]
        # order by decreasing k
        groups.sort(key=lambda i: i[1], reverse=True)
        for B in (1024, 2048):
            self._benchmark_rcr(
                B,
                groups,
                [],
                benchmark_non_group=True,
                test_name="17x_inline_cvr_11_groups",
            )

    def test_ads_11x_ctr_17_groups(self):
        for B in (1024, 2048):
            self._benchmark_batch_rcr(
                b=17,
                m=B,
                n=160,
                k=512,
                test_name="11x_ctr_17_groups",
            )


if __name__ == "__main__":
    unittest.main()
