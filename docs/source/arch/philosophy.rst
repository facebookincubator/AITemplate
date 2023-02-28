Design  Philosophy
==================


KISS (Keep it simple and stupid)
--------------------------------

AITemplate avoids deep IR lowering stacks to reduce the system's complexity.
A highly modularized, multiple backend codegen system written in pure Python directly attacks the pain point in high-performance GPU inference.

Pragmatism
----------

AITemplate provides a PyTorch-style frontend to enable engineers to manually match the PyTorch model & weights to AITemplate for optimization.
Using it is less painful than debugging different lowering IR stacks, especially for complex models such as MaskRCNN.

We believe most of the neural network workload can be decoupled.
For example, most of the network can be decoupled into Encoder, Decoder, and Decoder logics.
For encoder and decoder, it is a computation-bounded problem.
For decoder logic, it may involve more control flows.
By using divide and conquer, we left the decoder logic part to C++ or Python rather than build a unified language / IR stack as a silver bullet.
