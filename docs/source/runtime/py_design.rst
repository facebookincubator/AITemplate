===================
Python Runtime Note
===================

Python `Model`
==============

`Model` is a collection of Python bindings to the C++ AIT runtime. This section describes the API.

`AITData`
---------

This class represents a contiguous blob of memory that AIT will use as a tensor. It is simply a named tuple with these fields:

* `data_ptr: int`: An **unowned** pointer to **GPU** memory. In general, all of the APIs expect that this pointer will be valid for the entire duration of the call.
* `shape: List[int]`: The shape of the tensor.
* `dtype: str`: The tensor's dtype; one of `"float32", "float16", "int32", "int64"`. Note that most ops only support float16 at this stage.

When using AITemplate with PyTorch, `AITData` can be constructed with the `torch_to_ait_data` utility:

.. code-block:: python

      x = torch.randn(3, 3, 3).half().cuda()
      # Equivalent to AITData(x.data_ptr(), [3, 3, 3], "float16")
      x_ait = torch_to_ait_data(x)


If PyTorch is not available, `Model` provides a set of functions for copying, allocating, and freeing GPU memory. See the docstrings in `compiler/model.py` for more information.

`run`
-----

`run` takes inputs and outputs as collections of `AITData` instances. Both arguments can be passed as either an ordered list or a dictionary (mapping name to tensor).

.. code-block:: python

    # Arguments as a dictionary
    module.run(
      {"input0": in0_ait, "input1": in1_ait},
      {"output0": out0_ait, "output1": out1_ait},
    )

    # Arguments as an ordered list. Note that you might need to query
    # the index mapping.
    input_name_to_idx = module.get_input_name_to_index_map()
    output_name_to_idx = module.get_output_name_to_index_map()

    inputs = [None] * len(input_name_to_idx)
    outputs = [None] * len(output_name_to_idx)

    for name in input_name_to_idx:
      inputs[input_name_to_idx[name]] = ait_inputs[name]

    for name in output_name_to_idx:
      outputs[output_name_to_idx[name]] = ait_outputs[name]

    module.run(inputs, outputs)


One important caveat is that the output must have the **maximum** possible size. This is because of dynamic shapes: the size of the output may vary, but its shape is not inferred until inference time. The maximum shape can be queried with the `get_output_maximum_shape()`:

.. code-block:: python

    # Can use either name or index.
    name_to_idx = module.get_output_name_to_idx()
    max_shape = module.get_output_maximum_shape(name_to_idx["output"])
    max_shape = module.get_output_maximum_shape("output")


`Model.run` returns a dictionary of output `AITData` instances with (possibly dynamic) shapes that inferred in the runtime.

Nullptr Inputs/Outputs
----------------------

In general, inputs are allowed to be null if they are size 0 (e.g. at least one dimension is 0). The runtime enforces this with a check before any kernels are launched.

.. code-block:: cpp

    If (input_name == nullptr && dim0 * dim1 * … * dimN != 0) {
      throw std::runtime_error(“input_name cannot be null!”);
    }


This is convenient since torch.data_ptr() returns null for size zero tensors. The dynamic shape computation is skipped if the lower bound of the tensor’s size is positive.

Constants
---------

There are two types of constants in AIT; *bound* and *unbound* constants. A bound constant is known at compile time and may participate in constant folding. Bound constants are copied into GPU memory at model loading time. Values for bound constants may be provided by passing a dictionary (mapping constant name to AIT tensor) to `compile_model`.

Unbound constants, on the other hand, do not participate in constant folding and must be provided before running the model. These must be set via `Model.set_constant`:

.. code-block:: python

    module.set_constant("my_constant", AITData(...))
    # The pointer in the the tensor must live for the entire duration of run()
    module.run(...)


Constants are read-only and *shared* with all runtimes in the `ModelContainer`.

`run_with_tensors`
------------------

`run_with_tensors` is a convenience method with the same interface as `run`, except it can take lists (or dicts) of `torch.Tensor` instances:

.. code-block:: python

    input0 = torch.randn(input0_shape).cuda().half()
    output0 = torch.empty(output0_shape).cuda().half()
    # Returns a dictionary of reshaped outputs
    result = module.run_with_tensors([input0], [output0])


Streams and Asynchronous Predictions
------------------------------------

A pointer to a stream can optionally be passed to `run`.
If none is given, the prediction happens on the default stream 0.
If the `sync` argument is set to `True`, the stream is synchronized before `run()` returns.
`sync` is `True` by default.

Multiple predictions can happen at the same time (on the same or different streams).
Under the hood, there is a fixed-size pool of runtime objects.
When all the runtimes are used, `run()` blocks until one becomes available.
The size of this pool can be configured with the `num_runtimes` option in `Model`'s constructor.

CUDA Graph
----------

Run also takes a `graph_mode` option. If set to true, the runtime will try to use [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) to run the model. `graph_mode` is not supported on ROCm.

The following is a high level overview of how graph mode works:

1) Each `Model` has an internal stream used for graph capturing. The model first runs all ops on this stream in capture mode. No kernel launches happen during this stage.
2) If this is the first run, a graph is instantiated via `cudaGraphInstantiate`.
3) On subsequent runs, we try to avoid the relatively expensive `cudaGraphInstantiate` call by updating the graph executor (`cudaGraphExecUpdate`). However, a new graph may still be instantiated if the topology of the graph somehow changed between runs.
4) Once we have the graph executor, we launch a single kernel on the stream that the user provided to `run()`.

Graph mode is mainly beneficial when there are many small kernel launches. A lot of overhead can be avoided since there is only a single kernel launch in graph mode.
