# Docker + AITemplate

AITemplate provides docker image with tests, benchmark, documentation dependency installed. 

## Build CUDA Docker Image

Execute command 
```bash docker/build.sh cuda```
Will build a CUDA 11 docker image with tag: `ait:latest` 

## Build ROCM Docker Image

Execute command 
```bash docker/build.sh rocm```
Will build a RCOM 5 docker image with tag: `ait:latest` 

## Running Unit Tests in Docker

```docker run --gpus all ait:latest bash /AITemplate/tests/nightly/unittest.sh```

## Running Benchmark in Docker

```docker run --gpus all ait:latest bash /AITemplate/tests/nightly/benchmark.sh```
