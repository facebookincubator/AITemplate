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
import copy
import time
import unittest

import uuid
from enum import Enum
from typing import Callable, List, Optional, Set
from unittest import TestCase

import torch
from aitemplate.testing import detect_target
from fx2ait.acc_tracer import acc_tracer
from fx2ait.acc_tracer.ait_acc_normalizer import update_acc_op_mappers_for_ait
from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter
from fx2ait.tensor_spec import TensorSpec

OSS_AITModel = False
try:
    torch.ops.load_library("//deeplearning/ait:AITModel")
    print("===Load non-OSS AITModel===")
except Exception:
    torch.ops.load_library("build/libait_model.so")
    print("===Load OSS AITModel===")
    OSS_AITModel = True


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
class AITTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(3)
        detect_target()

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
        passes: List[Callable] = [],  # noqa: B006
        leaf_module: Callable = None,  # one leaf module
        apply_passes_to_lowered_module_only=False,
        use_fp16_acc=True,
    ):
        # TODO: add precision to interpreter once AIT supports multiple precision level
        # TODO: @qxy11 remove permute options once AIT supports channels-first format
        mod.eval()

        leaf_module_list = []
        if leaf_module:
            leaf_module_list.append(leaf_module)

        orig_mod = copy.deepcopy(mod)
        orig_mod.eval()
        mod = acc_tracer.trace(
            mod,
            inputs,
            leaf_module_list=leaf_module_list,
        )
        for p in passes:
            mod = p(mod, inputs)

        print(mod.graph)

        original_inputs = copy.deepcopy(inputs)
        if permute_inputs:
            inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]

        mod.half()
        inputs = [inp.half().contiguous() for inp in inputs]
        interp = AITInterpreter(
            mod,
            inputs,
            "/tmp",
            f"test-fx2ait-{uuid.uuid1()}",
            use_fp16_acc=use_fp16_acc,
        )
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if apply_passes_to_lowered_module_only:
                ref_outputs = orig_mod(*original_inputs)
            else:
                ref_outputs = mod(*original_inputs)
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)
            if unexpected_ops:
                self.assert_unexpected_op(mod, unexpected_ops)
            start = time.perf_counter()
            interp_result = interp.run()
            sec = time.perf_counter() - start
            print("Interpreter run time(s):", sec)
            if OSS_AITModel:
                ait_mod = AITModule(
                    torch.classes.ait.AITModel(
                        interp_result.engine.lib_path,
                        interp_result.input_names,
                        interp_result.output_names,
                        torch.float16,
                        torch.float,
                        1,  #  num_runtimes
                    ),
                    interp_result,
                )
            else:
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

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = ait_mod(*cuda_inputs)
            end_event.record()
            torch.cuda.synchronize()
            print("AIT run time(s)=", (start_event.elapsed_time(end_event) * 1.0e-3))
            # PyTorch Transformer model would yield 2 output tensors, of which the second one is
            # not useful. AIT model only output 1 output tensor, alter ref_output to match this.
            if leaf_module == torch.nn.MultiheadAttention:
                ref_outputs = ref_outputs[0]
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
    ):
        mod.eval()
        inputs_list = []
        for use_lower_bound in [True, False]:
            inputs_list.append(
                TensorSpec.create_inputs_from_specs(
                    inputs_spec, use_lower_bound=use_lower_bound
                )
            )

        inputs_min = inputs_list[0]
        inputs_max = inputs_list[1]
        mod.eval()
        mod = acc_tracer.trace(mod, inputs_min)
        original_inputs = inputs_min
        # Trace and test with inputs_min
        interp = AITInterpreter(
            mod,
            inputs_spec,
            "/tmp",
            f"test-fx2ait-{uuid.uuid1()}",
        )
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs_min:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)
            if unexpected_ops:
                self.assert_unexpected_op(mod, unexpected_ops)
            start = time.perf_counter()
            interp_result = interp.run()
            sec = time.perf_counter() - start
            print("Interpreter run time(s):", sec)
            if OSS_AITModel:
                ait_mod = AITModule(
                    torch.classes.ait.AITModel(
                        interp_result.engine.lib_path,
                        interp_result.input_names,
                        interp_result.output_names,
                        torch.float16,
                        torch.float,
                        1,  #  num_runtimes
                    ),
                    interp_result,
                )
            else:
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

            ref_outputs = mod(*original_inputs)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = ait_mod(*cuda_inputs)
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

                torch.testing.assert_close(
                    out.cpu(), ref, rtol=rtol, atol=atol, check_dtype=False
                )

            # To test dynamic shape, we test it again with inputs_max
            ref_outputs_max = mod(*inputs_max)
            for i in inputs_max:
                cuda_inputs_max = [i.cuda() for i in inputs_max]
            outputs_max = ait_mod(*cuda_inputs_max)
            if isinstance(outputs_max, torch.Tensor):
                ref_outputs_max = [ref_outputs_max]
                outputs_max = [outputs_max]
            for out, ref in zip(outputs_max, ref_outputs_max):
                if not isinstance(ref, torch.Tensor):
                    ref = torch.tensor([ref])
                ref = ref.cpu()  # to_dtype test has cases with gpu output

                torch.testing.assert_close(
                    out.cpu(), ref, rtol=rtol, atol=atol, check_dtype=False
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
    name: str,
    iters: int,
    mod: torch.nn.Module,
    inputs: List[torch.Tensor],
    permute_inputs: Optional[List[int]] = None,
) -> float:
    mod.eval()
    mod = acc_tracer.trace(
        mod,
        inputs,
    )
    original_inputs = inputs
    if permute_inputs:
        inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]
    interp = AITInterpreter(
        mod,
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


update_acc_op_mappers_for_ait()
