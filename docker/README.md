# Docker + AITemplate

AITemplate provides a Docker image with all test, benchmark, and documentation dependencies installed.

## Build CUDA Docker Image

```bash docker/build.sh cuda```
This will build a CUDA 11 docker image with tag: `ait:latest`

## Build ROCM Docker Image

```DOCKER_BUILDKIT=1 bash docker/build.sh rocm```
This will build a RCOM 5 docker image with tag: `ait:latest`

## Running Unit Tests in Docker

```docker run --gpus all ait:latest bash /AITemplate/tests/nightly/unittest.sh```

## Launching CUDA Docker
```docker run --gpus all -it ait:latest```

## Launching ROCM Docker

```docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ait:latest```


## Common questions:
- Q: When building ROCm Docker, I hit this error ` => ERROR [internal] load metadata for docker.io/library/ubuntu:20.04`, what shall I do?

  A: Run `docker pull docker.io/library/ubuntu:20.04` to pull base image manually, then re-run `./docker/build.sh rocm`
