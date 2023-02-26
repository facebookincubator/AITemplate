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

import uuid
from enum import Enum
from typing import Callable, List, Optional, Set
from unittest import TestCase

# executorch
import executorch.exir as exir
import torch
from aitemplate.compiler.public import DynamicProfileStrategy
from executorch.exir import CaptureConfig, ServerCompileConfig

from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter

from fx2ait.passes.lower_basic_pass_aten import (
    compose_bmm,
    compose_chunk,
    compose_getitem_slice,
    remove_ops,
    replace_aten_op_with_indices,
    replace_aten_reshape_alias_with_replace,
    # replace_batch_norm,  # it is needed if enable_aot=True in tracer
    replace_builtin_ops,
    replace_native_layernorm_with_layernorm,
    replace_transpose_mm_op_with_linear,
    run_const_fold,
)
from fx2ait.tensor_spec import TensorSpec

_LOGGER = logging.getLogger(__name__)
torch.ops.load_library("//deeplearning/ait:AITModel")


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


def fetch_attr(mod, target):
    """
    Fetch an attribute from the ``Module`` hierarchy of ``mod.module``.

    Args:
        target (str): The fully-qualfiied name of the attribute to fetch

    Return:
        Any: The value of the attribute.
    """
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class DispatchTestCase(TestCase):
    def generate_graph(
        self,
        mod: torch.nn.Module,
        original_inputs: List[torch.Tensor],
        expected_ops: Set[Callable],
        unexpected_ops: Optional[Set[Callable]] = None,
        customized_passes: List[Callable] = None,
    ):
        # Torchdynamo+aot proxytensor tracer
        # Below are common passes
        passes_list = [
            compose_bmm,
            compose_chunk,
            compose_getitem_slice,
            replace_aten_reshape_alias_with_replace,
            replace_aten_op_with_indices,
            replace_transpose_mm_op_with_linear,  # after compose_bmm
            replace_native_layernorm_with_layernorm,
            remove_ops,
            replace_builtin_ops,  # after replace_native_layernorm_with_layernorm
        ]
        # Combine with customized passes specific to any model
        if customized_passes:
            passes_list.extend(customized_passes)

        fx_module = exir.capture(
            mod,
            tuple(original_inputs),
            CaptureConfig(
                pt2_mode=True,
                enable_functionalization=False,
                enable_dynamic_shape=True,
            ),
        )._to_server(ServerCompileConfig(passes=passes_list))

        fx_module = run_const_fold(fx_module)
        _LOGGER.info(f"aten fx graph: {fx_module.graph}")

        if len(expected_ops):
            self.assert_has_op(fx_module, expected_ops)
        if unexpected_ops:
            self.assert_unexpected_op(fx_module, unexpected_ops)

        return fx_module

    def run_test(
        self,
        mod: torch.nn.Module,
        inputs: List[torch.Tensor],
        expected_ops: Set[Callable],
        unexpected_ops: Optional[Set[Callable]] = None,
        rtol: float = 1e-02,
        atol: float = 1e-02,
        precision: LowerPrecision = LowerPrecision.FP16,
        permute_inputs: Optional[List[int]] = None,
        permute_outputs: Optional[List[int]] = None,
        customized_passes: List[Callable] = None,
    ):
        mod.eval()
        original_inputs = inputs
        if permute_inputs:
            inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]

        fx_module = self.generate_graph(
            mod, original_inputs, expected_ops, unexpected_ops, customized_passes
        )

        interp = AITInterpreter(
            fx_module,
            inputs,
            "/tmp",
            f"test-aten2ait-{uuid.uuid1()}",
        )
        interp_result = interp.run()
        ait_mod_run = AITModule(
            torch.classes.fb.AITModel(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float,
                1,  #  num_runtimes
            ),
            interp_result,
        )

        # Inference run and results comparison
        with torch.no_grad():
            # reference run
            ref_outputs = mod(*original_inputs)
            # ait run
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = ait_mod_run(*cuda_inputs)
            end_event.record()
            torch.cuda.synchronize()
            print("AIT run time(s)=", (start_event.elapsed_time(end_event) * 1.0e-3))

            if isinstance(outputs, torch.Tensor):
                ref_outputs = [ref_outputs]
                outputs = [outputs]
            for out, ref in zip(outputs, ref_outputs):
                if not isinstance(ref, torch.Tensor):
                    ref = torch.tensor([ref])
                ref = ref.cpu()  # to_dtype test has cases with gpu output
                if permute_outputs:
                    out = out.permute(*permute_outputs)
                torch.testing.assert_close(
                    out.cpu(),
                    ref,
                    rtol=rtol,
                    atol=atol,
                    check_dtype=False,
                    equal_nan=True,
                )

    def run_test_with_dynamic_shape(
        self,
        mod: torch.nn.Module,
        inputs_spec: List[TensorSpec],
        expected_ops: Set[Callable],
        unexpected_ops: Optional[Set[Callable]] = None,
        rtol: float = 1e-02,
        atol: float = 1e-02,
        precision: LowerPrecision = LowerPrecision.FP16,
        permute_inputs: Optional[List[int]] = None,
        permute_outputs: Optional[List[int]] = None,
        customized_passes: List[Callable] = None,
        dynamic_profile_strategy=DynamicProfileStrategy.MAX,
        specify_num: Optional[float] = None,
    ):
        mod.eval()
        inputs_list = []
        for use_lower_bound in [True, False]:
            inputs_list.append(
                TensorSpec.create_inputs_from_specs(
                    inputs_spec,
                    use_lower_bound=use_lower_bound,
                    specify_num=specify_num,
                )
            )
        inputs = inputs_list[0]

        fx_module = self.generate_graph(
            mod, inputs, expected_ops, unexpected_ops, customized_passes
        )

        if permute_inputs:
            for inp in inputs_spec:
                shape = []
                for i in permute_inputs:
                    shape.append(inp.shape[i])
                inp.shape = shape

        interp = AITInterpreter(
            fx_module,
            inputs_spec,
            "/tmp",
            f"test-aten2ait-{uuid.uuid1()}",
            dynamic_profile_strategy=dynamic_profile_strategy,
        )
        interp_result = interp.run()
        ait_mod_run = AITModule(
            torch.classes.fb.AITModel(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float,
                1,  #  num_runtimes
            ),
            interp_result,
        )

        for inputs in inputs_list:
            with torch.no_grad():
                ref_outputs = mod(*inputs)
                # ait run
                cuda_inputs = []
                # reference run
                if permute_inputs:
                    inputs = [
                        inp.permute(*permute_inputs).contiguous() for inp in inputs
                    ]
                for i in inputs:
                    cuda_inputs.append(i.cuda())
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs = ait_mod_run(*cuda_inputs)
                end_event.record()
                torch.cuda.synchronize()
                print(
                    "AIT run time(s)=", (start_event.elapsed_time(end_event) * 1.0e-3)
                )

                if isinstance(outputs, torch.Tensor):
                    ref_outputs = [ref_outputs]
                    outputs = [outputs]
                for out, ref in zip(outputs, ref_outputs):
                    if not isinstance(ref, torch.Tensor):
                        ref = torch.tensor([ref])
                    ref = ref.cpu()  # to_dtype test has cases with gpu output
                    if permute_outputs:
                        out = out.permute(*permute_outputs)

                    torch.testing.assert_close(
                        out.cpu(),
                        ref,
                        rtol=rtol,
                        atol=atol,
                        check_dtype=False,
                        equal_nan=True,
                    )

    def assert_has_op(self, mod, ops):
        ops_in_mod = set()

        for node in mod.graph.nodes:
            if node.op == "call_module":
                ops_in_mod.add(type(fetch_attr(mod, node.target)))
            elif node.op in {"call_function", "call_method"}:
                ops_in_mod.add(node.target)

        self.assertTrue(
            ops_in_mod >= ops, f"expected ops {ops}, actuall ops {ops_in_mod}"
        )

    def assert_unexpected_op(self, mod, ops):
        for node in mod.graph.nodes:
            if node.op == "call_module":
                if type(fetch_attr(mod, node.target)) in ops:
                    return False
            elif node.op in {"call_function", "call_method"}:
                if node.target in ops:
                    return False
        return True

    def benchmark_function(
        self,
        name: str,
        iters: int,
        mod: torch.nn.Module,
        inputs: List[torch.Tensor],
        permute_inputs: Optional[List[int]] = None,
        customized_passes: Optional[List[int]] = None,
    ) -> float:
        mod.eval()
        original_inputs = inputs
        if permute_inputs:
            inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]

        fx_module = self.generate_graph(
            mod, original_inputs, {}, customized_passes=customized_passes
        )

        interp = AITInterpreter(
            fx_module,
            inputs,
            "/tmp",
            f"benchmark-fx2ait-{uuid.uuid1()}",
        )

        def benchmark(f, args):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            print("== Start benchmark iterations")
            with torch.inference_mode():
                start_event.record()
                for _ in range(iters):
                    f(*args)
                end_event.record()
            torch.cuda.synchronize()
            print("== End benchmark iterations")
            time_per_iter_ms = (start_event.elapsed_time(end_event) * 1.0e-3) / iters
            return time_per_iter_ms

        with torch.inference_mode():
            interp_result = interp.run()
            ait_mod = AITModule(
                torch.classes.fb.AITModel(
                    interp_result.engine.lib_path,
                    interp_result.input_names,
                    interp_result.output_names,
                    torch.float16,
                    torch.float,
                    1,  #  num_runtimes
                ),
                interp_result,
            )
            # Benchmark Pytorch Eager
            # warmup
            for _ in range(10):
                mod(*original_inputs)
            batch_size = inputs[0].shape[0]
            pt_time_per_iter_ms = benchmark(mod, original_inputs)
            pt_qps = batch_size / pt_time_per_iter_ms

            # Benchmark FX2AIT
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())
            # warmup
            for _ in range(10):
                ait_mod(*cuda_inputs)

            ait_time_per_iter_ms = benchmark(ait_mod, cuda_inputs)
            ait_qps = batch_size / ait_time_per_iter_ms

            result = (
                f"== Benchmark Result for: {name}\n"
                f"BS: {batch_size}, "
                f"PT Eager time per iter: {pt_time_per_iter_ms}ms, "
                f"PT Eager QPS: {pt_qps:.2f}, "
                f"FX2AIT time per iter: {ait_time_per_iter_ms}ms, "
                f"FX2AIT Eager QPS: {ait_qps:.2f}, "
                f"Speedup: {ait_qps/pt_qps:.2f}, "
            )
            with open("/tmp/bench_" + name + ".csv", "a") as f:
                f.write(
                    ",".join(
                        map(
                            str,
                            [
                                name,
                                batch_size,
                                pt_time_per_iter_ms,
                                pt_qps,
                                ait_time_per_iter_ms,
                                ait_qps,
                                ait_qps / pt_qps,
                            ],
                        )
                    )
                    + "\n"
                )
            return result
