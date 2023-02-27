
AITemplate Documentation
========================

AITemplate (AIT) is a Python framework that transforms deep neural networks into CUDA (NVIDIA GPU) / HIP (AMD GPU) C++ code for lightning-fast inference serving. AITemplate highlights include:

* High performance: close to roofline fp16 TensorCore (NVIDIA GPU) / MatrixCore (AMD GPU) performance on major models, including ResNet, MaskRCNN, BERT, VisionTransformer, Stable Diffusion, etc.
* Unified, open, and flexible. Seamless fp16 deep neural network models for NVIDIA GPU or AMD GPU. Fully open source, Lego-style easily extendable high-performance primitives for new model support. Supports a significantly more comprehensive range of fusions than existing solutions for both GPU platforms.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install/index


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   tutorial/index
   debughints

.. toctree::
   :maxdepth: 1
   :caption: Runtime Design

   runtime/index

.. toctree::
   :maxdepth: 1
   :caption: Architecture Guide

   arch/index


.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   reference/index
   reference/env
   genindex
