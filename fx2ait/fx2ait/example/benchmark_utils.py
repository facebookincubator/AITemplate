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
import time

import uuid
from typing import List, Optional

import torch
from fx2ait.acc_tracer import acc_tracer

from fx2ait.ait_module import AITModule

from fx2ait.fx2ait import AITInterpreter

torch.ops.load_library("build/libait_model.so")


def verify_accuracy(
    mod: torch.nn.Module,
    inputs: List[torch.Tensor],
    rtol: float = 1e-01,
    atol: float = 1e-01,
    permute_inputs: Optional[List[int]] = None,
    permute_outputs: Optional[List[int]] = None,
):
    # TODO: add precision to interpreter once AIT supports multiple precision level
    # TODO: @qxy11 remove permute options once AIT supports channels-first format
    mod.eval()
    mod = acc_tracer.trace(
        mod,
        inputs,
    )
    print(mod)

    original_inputs = inputs
    if permute_inputs:
        inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]
    interp = AITInterpreter(
        mod,
        inputs,
        "/tmp",
        f"test-fx2ait-{uuid.uuid1()}",
    )
    with torch.no_grad():
        cuda_inputs = []
        for i in inputs:
            cuda_inputs.append(i.cuda())

        mod.eval()

        start = time.perf_counter()
        interp_result = interp.run()
        sec = time.perf_counter() - start
        print("Interpreter run time(s):", sec)
        ait_mod = AITModule(
            torch.classes.ait.AITModel(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float,
                1,  #  num_runtimes
            )
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
            if permute_outputs:
                out = out.permute(*permute_outputs)
            print(out)
            print(ref)
            torch.testing.assert_close(
                out.cpu(),
                ref,
                rtol=rtol,
                atol=atol,
                check_dtype=False,
                equal_nan=True,
            )


def benchmark_function(
    name: str,
    iters: int,
    mod: torch.nn.Module,
    inputs: List[torch.Tensor],
    permute_inputs: Optional[List[int]] = None,
    ait_mod: torch.nn.Module = None,
) -> float:
    mod.eval()
    original_inputs = inputs
    if permute_inputs:
        inputs = [inp.permute(*permute_inputs).contiguous() for inp in inputs]

    if ait_mod is None:
        mod = acc_tracer.trace(
            mod,
            original_inputs,
        )
        interp = AITInterpreter(
            mod,
            inputs,
            "/tmp",
            f"benchmark-fx2ait-{uuid.uuid1()}",
        )

        interp_result = interp.run()
        ait_mod = AITModule(
            torch.classes.ait.AITModel(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float,
                1,  #  num_runtimes
            )
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
