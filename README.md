# AITemplate

[![License](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE) |
[![Documentation](https://github.com/facebookincubator/AITemplate/actions/workflows/docs.yml/badge.svg)](https://facebookincubator.github.io/AITemplate) |
[![CircleCI](https://circleci.com/gh/facebookincubator/AITemplate.svg?style=svg)](https://app.circleci.com/pipelines/github/facebookincubator/AITemplate)




AITemplate (AIT) is a Python framework that transforms deep neural networks into CUDA (NVIDIA GPU) / HIP (AMD GPU) C++ code for lightning-fast inference serving. AITemplate highlights include:

- High performance: close to roofline fp16 TensorCore (NVIDIA GPU) / MatrixCore (AMD GPU) performance on major models, including ResNet, MaskRCNN, BERT, VisionTransformer, Stable Diffusion, etc.
- Unified, open, and flexible. Seamless fp16 deep neural network models for NVIDIA GPU or AMD GPU. Fully open source, Lego-style easy extendable high-performance primitives for new model support. Supports a significantly more comprehensive range of fusions than existing solutions for both GPU platforms.

## More about AITemplate

### Excellent Backward Capability

AITemplate doesn't depend on third-party libraries or runtimes, such as cuBLAS, cuDNN, rocBLAS, MIOpen, TensorRT, MIGraphX, etc. Each model is compiled into a self-contained portable binary, which can be used on any software environment with the same hardware.

### Horizontal Fusion

AITemplate provides unique advanced horizontal fusion. AITemplate can fuse parallel GEMM, LayerNorm, and other operators with different input shapes into a single GPU kernel.

### Vertical Fusion

AITemplate provides strong vertical fusion. AITemplate can fuse a large range of operations into TensorCore/MatrixCore operations, such as elementwise operations, reduction operations, and layout permutation operations. AITemplate also provides back-to-back style TensorCore / MatrixCore operation fusion.

### Memory Fusion

AITemplate provides innovative memory fusions. AITemplate can fuse GEMM, LayerNorm, and other operators, followed by memory operations such as concatenation, split, and slice into a single operator.

### Working w/wo PyTorch
The AITemplate-generated Python runtime can take PyTorch tensors as inputs and outputs without an extra copy. For environments without PyTorch, the AITemplate Python/C++ runtime is self-contained.

### Extensions without suffering

AITemplate provides a straightforward approach for making an extension in codegen. To add a new operator or a new fused kernel into AITemplate, most of the time one only needs to add two Python files: one for a graph node definition and another for the backend codegen. The CUDA/HIP kernel in a text header file can be directly utilized in the codegen.

## Installation

**Hardware requirement:**
  - **NVIDIA**: AIT is only tested on SM80+ GPUs (Ampere etc). Not all kernels work with old SM75/SM70 (T4/V100) GPUs.
  - **AMD**:  AIT is only tested on CDNA2 (MI-210/250) GPUs. There may be compiler issues for old CDNA1 (MI-100) GPUs.

## Clone the code
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
AITemplate provides the following model templates & reference performance data on A100/MI-250

- [01_ResNet-50](examples/01_resnet-50/) with PyTorch Image Models (TIMM)
- [02_MaskRCNN-FPN](examples/02_detectron2/) with Detectron2
- [03_BERT](examples/03_bert/) with HuggingFace Transformer
- [04_Vision Transformer](examples/04_vit/) with PyTorch Image Models (TIMM)
- [05_Stable Diffusion](examples/05_stable_diffusion/) with HuggingFace Diffusers

## Release

AITemplate has a 90 days release cycle.
In the next one or two releases, we will focus on:
- Deprecating FlashAttention: Unify CUDA Attention computation to Composable Kernel (AMD GPU) style back-to-back fusion to improve performance and increase flexibility for NVIDIA GPU Transformer users.
- Remove kernel profiling requirement.
- GEMM + LayerNorm fusion, GEMM + GEMM fusion, Conv + Conv fusion.
- Better dynamic shape support: Focus on the dynamic sequence in Transformers.
- More model templates:  Provide model templates with control flow and containers.
- More automatic graph passes: Relief manual rewrite models to obtain the best performance.
- Enable more fusions on AMD backend.

Some ongoing/potential work that won't appear in the next short-term release:
- Automatic Pytorch-FX, ONNX, Open-XLA and other format model conversion.
- Quantized model (int8/fp8/int4) support.
- Composable Kernel CPU extension on AVX2/AVX-512 for AMD Epyc CPU.

## Contributing
Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute to the project.

## The Team

AITemplate is co-created by Meta engineers: [Bing Xu](https://github.com/antinucleon), [Ying Zhang](https://github.com/ipiszy), [Hao Lu](https://github.com/hlu1), [Yang Chen](https://github.com/chenyang78), and [Terry Chen](https://github.com/terrychenism), with major contributions coming from more talented engineers. A non-exhaustive list to mention is Mike Iovine, Mu-Chu Lee, Scott Wolchok, Oleg Khabinov, Shirong Wu, Huaming Li, Hui Guo, Zhijing Li, Max Podkorytov. We also want to thank the discussions with Andrew Tulloch, Yinghai Lu, Lu Fang.

AITemplate is currently maintained by Meta engineers: [Ying Zhang](https://github.com/ipiszy), [Hao Lu](https://github.com/hlu1), [Yang Chen](https://github.com/chenyang78), [Terry Chen](https://github.com/terrychenism), [Mike Iovine](https://github.com/mikeiovine), [Mu-Chu Lee](https://github.com/muchulee8) and [Bing Xu](https://github.com/antinucleon).


## Acknowledgement

AITemplate team works deeply with NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass) Team (Led by Andrew Kerr, Haicheng Wu) and AMD [Composable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) Team (Led by Chao Liu, Jing Zhang). We co-designed many advanced GPU optimizations specialized for each platform, and nothing is possible without our close collaboration.


## License
AITemplate is licensed under the [Apache 2.0 License](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE).
