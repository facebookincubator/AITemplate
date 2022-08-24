# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] helper function to benchmark fx-trt
"""
# pylint: disable=C0415,R0402,C0103


def benchmark_trt_function(iters: int, function, *args) -> float:
    """[summary]

    Parameters
    ----------
    iters : int
        [description]
    function : [type]
        [description]

    Returns
    -------
    float
        [description]
    """
    import tensorrt as trt

    from .detect_target import _detect_fb

    if _detect_fb():
        import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
        from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
    else:
        import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
        from torch.fx.experimental.fx2trt import (
            InputTensorSpec,
            TRTInterpreter,
            TRTModule,
        )

    from .benchmark_pt import benchmark_torch_function

    inputs = args
    acc_mod = acc_tracer.trace(function, inputs)
    input_specs = InputTensorSpec.from_tensors(inputs)
    interpreter = TRTInterpreter(
        acc_mod,
        input_specs,
        logger_level=trt.Logger.INFO,
        explicit_batch_dimension=False,
    )
    result = interpreter.run(
        max_batch_size=256,
        max_workspace_size=4 << 30,
        fp16_mode=True,
        profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        # strict_type_constraints=True,
    )

    submod = TRTModule(result.engine, result.input_names, result.output_names)
    submod(*inputs)
    t2 = benchmark_torch_function(
        iters,
        lambda: submod(*inputs),
    )
    return t2
