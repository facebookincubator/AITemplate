Installing AITemplate
=====================

Using Docker
------------

The easiest way to get started is to use Docker.  Using docker is able to avoid performance regression caused by incorrect version of NVCC and HIPCC.
To use docker, we provide a bash script to build the docker image.

- CUDA:
    .. code-block:: bash

        ./docker/build.sh cuda
- ROCM:
    .. code-block:: bash

        DOCKER_BUILDKIT=1 ./docker/build.sh rocm


This will build a docker image with tag `ait:latest`.

To launch the docker container

- CUDA:
    .. code-block:: bash

        docker run --gpus all -it ait:latest

- ROCM:
    .. code-block:: bash

        docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ait:latest

AITemplate will be installed as a Python package in Python 3.8. There will be also a copy of the source code and examples at `/AITemplate`.


Installing as a Standard Python Package
---------------------------------------

Before installing AITemplate, first make sure you have correct hardware and software environment.

- Hardware
    - NVIDIA: AIT is only tested on SM80+ GPUs (Ampere etc).
    - AMD: AIT is only tested on CDNA2 (MI-210/250) GPUs.

.. warning::
    - Not all kernels work with old SM75/SM70 (T4/V100) GPUs.
    - There may be compiler issues for old CDNA1 (MI-100) GPUs.

- Software
    - NVIDIA: CUDA 11.6
    - AMD: ROCm 5.2, with HIPCC 10736 (commit `b0f4678b9058a4ae00200dfb1de0da5f2ea84dcb`)

.. warning::
    - Incorrect compiler version may lead to performance regression.
    - Instruction for building HIPCC 10736 can be founded in `docker/Dockerfile.rocm`.


When cloning the code, please use the following command to clone the submodules:

    .. code-block:: bash

        git clone --recursive https://github.com/facebookincubator/AITemplate

.. warning::
    Please check that all submodules are cloned correctly before the next step.

Then build the Python wheel package and install it:

    .. code-block:: bash

        cd python
        python setup.py bdist_wheel
        pip install dist/aitemplate-0.0.1-py3-none-any.whl
