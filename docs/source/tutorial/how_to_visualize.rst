How to visualize an AIT model
=============================

Visualization is important for understanding the behavior of a model optimization.
In AIT, we modify the codegen a little bit, from generating CUDA/HIP C++ code to HTML/Javascript code,
then we can generate a visualization of the model.


The following code will generate a visualization of our first example.

1. Define the AIT Model
-----------------------

.. code-block:: python

  from aitemplate import compiler
  from aitemplate.frontend import nn, Tensor
  from aitemplate.testing import detect_target
  from aitemplate.utils.visualization import plot_graph

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

  def gen_ait_model():
    batch_size = 512
    hidden = 1024
    ait_model = AITSimpleModel(hidden)
    ait_model.name_parameter_tensor()
    X = Tensor(
              shape=[batch_size, hidden],
              name="X",
              dtype="float16",
              is_input=True,
    )
    Y = ait_model(X)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    return Y

  output_tensor = gen_ait_model()

2. Apply optimizations on the AIT Model
---------------------------------------

.. code-block:: python

  def apply_optimizations(tensors):
    target = detect_target()
    # first, convert output tensors to graph
    with target:
      graph = compiler.transform.toposort(tensors)
      # second, provide names to the graph
      compiler.transform.name_graph(graph)
      compiler.transform.mark_param_tensor(graph)
      compiler.transform.mark_special_views(graph)
      # we can apply optimizations to the graph, or test single optimization pass on the graph
      graph = compiler.transform.optimize_graph(graph, "./tmp")
    return graph

  graph = apply_optimizations(output_tensor)

3. Generate visualization
-------------------------

.. code-block:: python

  # Plot the graph
  plot_graph(graph, file_path="ait_model.html")

The visualization will be generated in the "ait_model.html" file. This file can be opened in Chrome without any web server.

.. raw:: html

  <iframe src="ait_model.html" width="100%" height="600px"></iframe>
