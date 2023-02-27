# AITemplate

[![License](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE) |
[![Documentation](https://github.com/facebookincubator/AITemplate/actions/workflows/docs.yaml/badge.svg)](https://facebookincubator.github.io/AITemplate) |
[![CircleCI](https://circleci.com/gh/facebookincubator/AITemplate.svg?style=svg)](https://app.circleci.com/pipelines/github/facebookincubator/AITemplate)
[![Deploy docs to Pages](https://github.com/facebookincubator/AITemplate/actions/workflows/pages.yaml/badge.svg)](https://github.com/facebookincubator/AITemplate/actions/workflows/pages.yaml)


AITemplate (AIT) is a Python framework that transforms deep neural networks into CUDA (NVIDIA GPU) / HIP (AMD GPU) C++ code for lightning-fast inference serving. AITemplate highlights include:

- High performance: close to roofline fp16 TensorCore (NVIDIA GPU) / MatrixCore (AMD GPU) performance on major models, including ResNet, MaskRCNN, BERT, VisionTransformer, Stable Diffusion, etc.
- Unified, open, and flexible. Seamless fp16 deep neural network models for NVIDIA GPU or AMD GPU. Fully open source, Lego-style easily extendable high-performance primitives for new model support. Supports a significantly more comprehensive range of fusions than existing solutions for both GPU platforms.


## More about AITemplate

### Excellent Backward Capability

AITemplate doesn't depend on third-party libraries or runtimes, such as cuBLAS, cuDNN, rocBLAS, MIOpen, TensorRT, MIGraphX, etc. Each model is compiled into a self-contained portable binary, which can be used on any software environment with the same hardware.

### Horizontal Fusion

AITemplate provides unique advanced horizontal fusion. AITemplate can fuse parallel GEMM, LayerNorm, and other operators with different input shapes into a single GPU kernel.

### Vertical Fusion

AITemplate provides strong vertical fusion. AITemplate can fuse a large range of operations into TensorCore/MatrixCore operations, such as elementwise operations, reductions, and layout permutations. AITemplate also provides back-to-back style TensorCore / MatrixCore operation fusion.

### Memory Fusion

AITemplate provides innovative memory fusions. AITemplate can fuse GEMM, LayerNorm, and other operators, followed by memory operations such as concatenation, split, and slice into a single operator.

### Working w/wo PyTorch

The AITemplate-generated Python runtime can take PyTorch tensors as inputs and outputs without an extra copy. For environments without PyTorch, the AITemplate Python/C++ runtime is self-contained.

### Extensions without suffering

AITemplate provides a straightforward approach for making an extension in codegen. To add a new operator or a new fused kernel into AITemplate, most of the time one only needs to add two Python files: one for a graph node definition and another for the backend codegen. The CUDA/HIP kernel in a text header file can be directly utilized in the codegen.


## FX2AIT

FX2AIT is a Python-based tool that converts PyTorch models into AITemplate (AIT) engine for lightning-fast inference serving. Using FX2AIT's built-in AITLowerer, partial AIT acceleration can be achieved for models with unsupported operators in AITemplate.

Key features of FX2AIT include:

* Easy Conversion: FX2AIT requires only a PyTorch model and input for conversion, generating an "AITModule" output for inference serving.
* Expanded Support: AITemplate does not support all PyTorch operators. FX2AIT's AITLowerer offers a solution for partial AIT conversion for models with unsupported operators. Check the `fx2ait/fx2ait/example/03_lowering_split` for more information.

More info can be found from https://github.com/facebookincubator/AITemplate/tree/main/fx2ait.


## Installation

**Hardware requirements:**
  - **NVIDIA**: AIT is only tested on SM80+ GPUs (Ampere etc). Not all kernels work with old SM75/SM70 (T4/V100) GPUs.
  - **AMD**:  AIT is only tested on CDNA2 (MI-210/250) GPUs. There may be compiler issues for old CDNA1 (MI-100) GPUs.

### Clone the code

When cloning the code, please use the following command to also clone the submodules:
```
git clone --recursive https://github.com/facebookincubator/AITemplate
```

### Docker Image

We highly recommend using AITemplate with Docker to avoid accidentally using a wrong version of NVCC or HIPCC.
- CUDA: `./docker/build.sh cuda`
- ROCM: `DOCKER_BUILDKIT=1 ./docker/build.sh rocm`

This will build a docker image with tag `ait:latest`.

### From Source

The following command will create a Python wheel for AITemplate. Please ensure you have correct CUDA/ROCm compiler installed.
- CUDA: CUDA 11.6
- ROCm: We tested on ROCm 5.2.3 with a customized build HIPCC with the command in docker/Dockerfile.rocm#L87-L96

*Incorrect compiler will lead performance regression.*

**Please check all submodules are cloned correctly before go to next step.**

```
cd python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```

## Getting Started

Check out the [AITemplate Documentation](https://facebookincubator.github.io/AITemplate) for API reference.

There are a few tutorials for onboarding:

- 01: [How to inference a PyTorch model with AIT](https://facebookincubator.github.io/AITemplate/tutorial/how_to_infer_pt.html)
- 02: [How to add an op to AIT codegen](https://facebookincubator.github.io/AITemplate/tutorial/how_to_add_op.html)
- 03: [How to visualize AIT's optimization](https://facebookincubator.github.io/AITemplate/tutorial/how_to_visualize.html)


## Examples & Performance

AITemplate provides the following model templates & reference performance data on A100/MI-250:

- [01_ResNet-50](examples/01_resnet-50/) with PyTorch Image Models (TIMM)
- [02_MaskRCNN-FPN](examples/02_detectron2/) with Detectron2
- [03_BERT](examples/03_bert/) with HuggingFace Transformer
- [04_Vision Transformer](examples/04_vit/) with PyTorch Image Models (TIMM)
- [05_Stable Diffusion](examples/05_stable_diffusion/) with HuggingFace Diffusers

## Release

All current development updates can be seen in the AITemplate repository. Releases are not on a set schedule and will only be tagged for significant feature releases.

Mid-term plan:
- Better dynamic shape support: Focus on the dynamic sequence in Transformers. Add symbolic shape support.
- More automatic graph passes: Relief manual rewrite models to obtain the best performance.
- Quantization: fp8/int8/int4.
- Sparsity pruning for Gemm.
- PT2 integration: Aten2AIT is under active development.

Long-term plan:
- Automatic ONNX, Open-XLA and other format model conversion.
- Composable Kernel CPU extension on AVX2/AVX-512 for AMD Epyc CPU.

## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute to the project.

## The Team

AITemplate is currently maintained by Meta engineers: [Ying Zhang](https://github.com/ipiszy), [Yang Chen](https://github.com/chenyang78), [Terry Chen](https://github.com/terrychenism), [Mu-Chu Lee](https://github.com/muchulee8), [Max Podkorytov](https://github.com/tenpercent), [Adnan Akhundov](https://github.com/aakhundov).

AITemplate is co-created by Meta engineers: [Bing Xu](https://github.com/antinucleon), [Ying Zhang](https://github.com/ipiszy), [Hao Lu](https://github.com/hlu1), [Yang Chen](https://github.com/chenyang78), and [Terry Chen](https://github.com/terrychenism), with major contributions coming from more talented engineers. A non-exhaustive list to mention is Mike Iovine, Mu-Chu Lee, Scott Wolchok, Oleg Khabinov, Shirong Wu, Huaming Li, Hui Guo, Zhijing Li, Max Podkorytov. We also want to thank Andrew Tulloch, Yinghai Lu, Lu Fang for the valuable discussions.

FX2AIT and Aten2AIT are co-created and maintained by Meta engineers: [Wei Wei](https://github.com/frank-wei), [Shirong Wu](https://github.com/wushirong) and [Zhijing Li](https://github.com/tissue3).


## Acknowledgements

AITemplate team works deeply with NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass) Team (led by Andrew Kerr, Haicheng Wu) and AMD [Composable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) Team (led by Chao Liu, Jing Zhang). We co-designed many advanced GPU optimizations specialized for each platform, and nothing is possible without our close collaboration.


## License

AITemplate is licensed under the [Apache 2.0 License](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE).
