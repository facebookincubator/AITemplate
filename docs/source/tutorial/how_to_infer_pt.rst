How to inference a PyTorch model with AIT
=========================================

This tutorial will demonstrate how to inference a PyTorch model with AIT.
Full source code can be found at `examples/07_how_to_run_pt_model/how_to_run_pt_model.py`.

0. Prerequisites
----------------

We need to import necessary Python modules:

.. code-block:: python

  from collections import OrderedDict

  import torch

  from aitemplate.compiler import compile_model
  from aitemplate.frontend import nn, Tensor
  from aitemplate.testing import detect_target
  from aitemplate.testing.benchmark_pt import benchmark_torch_function
  from aitemplate.utils.graph_utils import sorted_graph_pseudo_code


1. Define a PyTorch module
--------------------------

Here we define a PyTorch model which is commonly seen in Transformers:

.. code-block:: python

  class PTSimpleModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
      super().__init__()
      self.dense1 = torch.nn.Linear(hidden, 4 * hidden)
      self.act1 = torch.nn.functional.gelu
      self.dense2 = torch.nn.Linear(4 * hidden, hidden)
      self.layernorm = torch.nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
      hidden_states = self.dense1(input)
      hidden_states = self.act1(hidden_states)
      hidden_states = self.dense2(hidden_states)
      hidden_states = hidden_states + input
      hidden_states = self.layernorm(hidden_states)
      return hidden_states

2. Define an AIT module
-----------------------

We can define a similar AIT module as follows:

.. code-block:: python

  class AITSimpleModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
      super().__init__()
      self.dense1 = nn.Linear(hidden, 4 * hidden, specialization="fast_gelu")
      self.dense2 = nn.Linear(4 * hidden, hidden)
      self.layernorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, input):
      hidden_states = self.dense1(input)
      hidden_states = self.dense2(hidden_states)
      hidden_states = hidden_states + input
      hidden_states = self.layernorm(hidden_states)
      return hidden_states

.. warning::
  The `nn.Module` API in AIT looks similar to PyTorch, but it is not the same.

  The fundamental difference is that AIT module is a container to build a graph, while PyTorch module is a container to store parameters for eager.
  Which means, each AIT module's `forward` method can be only called once, and the graph is built during the first call.
  If you want to share parameters, you need to use the `compiler.ops` instead. The `compiler.ops` is similar to `functional` in PyTorch.

  AITemplate supports automatic fusion of linear followed by other operators. However in many cases, especially for quick iterations, we use manual `specialization` to specify the fused operator. For example, `specialization="fast_gelu"` will fuse linear with the `fast_gelu` operator.

3. Define a helper function to map PyTorch parameters to AIT parameters
-----------------------------------------------------------------------

In AIT, all names must follow the C variable naming standard, because the names will be used in the codegen process.

.. code-block:: python

  def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
      ait_name = name.replace(".", "_")
      assert name in pt_params
      mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params

.. warning::

  - Different to PyTorch, it is required to call ait_model **.name_parameter_tensor()** method to provide each parameter with a name with a direct map to PyTorch.
  - Because all names in AIT must follow the C variable naming standard, you can easily replace `.` by `_` or use a regular expression to make sure the name in valid.
  - For networks with conv + bn subgraph, we currently don't provide an automatic pass to fold it. Please refer to our ResNet and Detectron2 examples to see how we handle CNN layout transform and BatchNorm folding.

4. Create PyTorch module, inputs/outputs
----------------------------------------

.. code-block:: python

  batch_size=1024
  hidden=512
  # create pt model
  pt_model = PTSimpleModel(hidden).cuda().half()

  # create pt input
  x = torch.randn([batch_size, hidden]).cuda().half()

  # run pt model
  pt_model.eval()
  y_pt = pt_model(x)

5. Create AIT module, inputs/outputs
------------------------------------

.. code-block:: python

  batch_size=1024
  hidden=512
  # create AIT model
  ait_model = AITSimpleModel(hidden)
  # create AIT input Tensor
  X = Tensor(
        shape=[batch_size, hidden],
        name="X",
        dtype="float16",
        is_input=True,
  )
  # run AIT module to generate output tensor
  Y = ait_model(X)
  # mark the output tensor
  Y._attrs["is_output"] = True
  Y._attrs["name"] = "Y"

.. warning::

  - Similar to MetaTensor, LazyTensor and a lot of other lazy evaluation frameworks, AIT's Tensor records the computation graph, and the graph is built when the Tensor is compiled.
  - For input tensor, it is required to set the attribute **is_input=True**.
  - For output tensor, it is required to set the attribute **Y._attrs["is_output"] = True**.
  - For input and output tensors, it is better to provide the **name** attributes to use in runtime.

6. Compile AIT module into runtime and do verification
------------------------------------------------------

.. code-block:: python

  # map pt weights to ait
  weights = map_pt_params(ait_model, pt_model)

  # codegen
  target = detect_target()
  with compile_model(
      Y, target, "./tmp", "simple_model_demo", constants=weights
  ) as module:
    # create storage for output tensor
    y = torch.empty([batch_size, hidden]).cuda().half()

    # inputs and outputs dict
    inputs = {"X": x}
    outputs = {"Y": y}

    # run
    module.run_with_tensors(inputs, outputs, graph_mode=True)

    # verify output is correct
    print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    # benchmark ait and pt
    count = 1000
    ait_t, _, _ = module.benchmark_with_tensors(
        inputs, outputs, graph_mode=True, count=count
    )
    print(f"AITemplate time: {ait_t} ms/iter")

    pt_t = benchmark_torch_function(count, pt_model.forward, x)
    print(f"PyTorch eager time: {pt_t} ms/iter")


In this example, AIT will automatically fuse GELU and elementwise addition into the TensorCore/MatrixCore gemm operation. On RTX-3080, in the example AIT is about 1.15X faster than PyTorch Eager.

.. note::

  - In this example, we fold the parameters (`weights`) into AIT runtime. The final dynamic library will contain them as parameters.
  - If during the compile time we don't provide the parameters (for example, because the total parameters size is greater than 2GB), we can always call `set_constant` function in the runtime. Please check the runtime API for the details.
