# FX2AIT for AITemplate


FX2AIT is a Python-based tool that converts PyTorch models into AITemplate (AIT) engine for lightning-fast inference serving.
AITLowerer built on top of FX2AIT is able to perform AIT conversion on PyTorch model with AIT unsupported operators. Model can enjoy partial AIT acceleration using AITLowerer.

FX2AIT highlights include:

- Automatic Conversion: FX2AIT only need PyTorch model and input as input for conversion. The output can be used for inference serving directly.
- Expanded Support: AITemplate doesn't cover all operators PyTorch provides. FX2AIT provided AITLowerer as solution to support partial AIT conversion for models with AIT unsupported operators. For more information, please check example/03_lowering_split.

## Installation

**Hardware requirement:**
  - **NVIDIA**: FX2AIT is based on AIT, thus the hardware requirement is same as AIT. AIT is only tested on SM80+ GPUs (Ampere etc). Not all kernels work with old SM75/SM70 (T4/V100) GPUs.
### From Source
The following command will create a Python wheel for AITemplate. Please ensure you have correct CUDA compiler installed.
- CUDA: CUDA 11.6
- cuDNN: v8.7.0 for CUDA 11.x
  download source: https://developer.nvidia.com/rdp/cudnn-download
  installation guidance: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

*Incorrect compiler will lead performance regression.*

```
cd fx2ait
python setup.py install
```

### Docker Image
We highly recommend using AITemplate with Docker to avoid accidentally using a wrong version of NVCC or HIPCC.
- CUDA: `./docker/build.sh cuda`

This will build a docker image with tag `ait:latest`.

## Examples
AITemplate provides the following getting started tutorials:
- 01: [How to inference a PyTorch Transformer model with FX2AIT](fx2ait/example/01_transformer_model/)
- 02: [How to inference a PyTorch vision model with FX2AIT](fx2ait/example/02_vision_model/)
- 03: [How to inference a general PyTorch model with AIT unsupported operator using AIT Lowerer](fx2ait/example/03_lowering_split/)
### Run Example and Test
Example command:
```
cd fx2ait
python example/03_lowering_split/test_lower.py
python test/test_ait_lower.py
```
