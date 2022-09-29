# ResNet-50

In this example, we will demo how to use AITemplate for inference on the ResNet-50 model from PyTorch Image Models (TIMM).

We will demo two usages:
* Using AIT to accelerate PyTorch inference
* Using AIT standalone without PyTorch

## Code structure
```
modeling
    resnet.py              # ResNet definition using AIT's frontend API
weight_utils.py            # Utils to convert TIMM R-50 weights to AIT
infer_with_torch.py        # Example to accelerate PyTorch, and seamlessly use with other PyTorch code
infer_with_numpy.py        # Dump TIMM weights to Numpy and use AIT & Numpy without 3rdparties
benchmark_pt.py            # Benchmark code for PyTorch
benchmark_ait.py           # Benchmark code for AIT
```

## Multi-GPU profiling
AIT requires to do profiling to decide best algorithms for CUTLASS and CK.
To enable multiple GPUs profiling, use the environment variable `CUDA_VISIBLE_DEVICES` on NVIDIA platform and `HIP_VISIBLE_DEVICES` on AMD platform.

For example, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_ait.py`.

Benchmark is fast once the profilers are built.

## Reference Speed vs PyTorch Eager

### A100-40GB / CUDA 11.6.2
_PT = PyTorch 1.12 Eager_

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 7.68            | 130.29        | 0.58             | 1730.17        |
| 2          | 7.16            | 279.36        | 0.62             | 3250.74        |
| 4          | 7.17            | 557.68        | 0.69             | 5773.20        |
| 8          | 7.02            | 1138.83       | 0.88             | 9104.44        |
| 16         | 7.01            | 2280.97       | 1.33             | 12012.81       |
| 32         | 7.53            | 4251.30       | 2.40             | 13350.58       |
| 64         | 13.98           | 4578.09       | 4.53             | 14140.83       |
| 128        | 26.57           | 4816.71       | 8.57             | 14935.82       |
| 256        | 50.93           | 5026.40       | 16.58            | 15444.57       |


### MI-250 / ROCm 5.2.3 / HIPCC-10736
_PT = PyTorch 1.12 Eager_
#### 1 GCD

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 3.94            | 254.06        | 2.28             | 438.60         |
| 2          | 3.89            | 514.48        | 2.25             | 888.89         |
| 4          | 3.82            | 1047.11       | 2.38             | 1680.67        |
| 8          | 4.40            | 1819.27       | 2.62             | 3053.44        |
| 16         | 6.48            | 2468.65       | 3.41             | 4692.08        |
| 32         | 10.40           | 3076.97       | 4.86             | 6584.36        |
| 64         | 18.35           | 3488.12       | 8.26             | 7748.18        |
| 128        | 34.36           | 3724.76       | 15.38            | 8322.50        |
| 256        | 65.35           | 3917.29       | 29.62            | 8642.81        |

#### 2 GCDs

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          |                 |               |                  |                |
| 2          | 3.94            | 507.54        | 2.36             | 848.15         |
| 4          | 3.89            | 1028.60       | 2.34             | 1710.94        |
| 8          | 3.88            | 2059.41       | 2.70             | 2960.46        |
| 16         | 4.56            | 3507.48       | 2.83             | 5663.52        |
| 32         | 6.72            | 4762.89       | 3.87             | 8275.98        |
| 64         | 10.82           | 5917.63       | 5.26             | 12173.67       |
| 128        | 18.79           | 6812.09       | 8.98             | 14247.09       |
| 256        | 35.99           | 7112.59       | 16.69            | 15338.58       |



### Note for Performance Results

- For NVIDIA A100, our test cluster doesn't allow to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- To benchmark MI-250, the first step is to run `python3 benchmark_ait.py` to generate all necessary model dynamic library files with single GCD. Then run `./benchmark_mi250.sh {batch_size}` to simulate data parallel execution on 2 GCDs, each GCD is processing half of the batch.
- To benchmark MI-250 1 GCD, we lock the frequency with command `rocm-smi -d x --setperfdeterminism 1700`, where `x` is the GPU id.
- To benchmark MI-250 2 GCDs, we observed performance regression with rocm perf-determ mode. The 2 GCDs number is running without perf-determ mode set with command `rocm-smi -d x --resetperfdeterminism`, where `x` is the GPU id.
- Performance results are what we can reproduce and for reference only. It should not be used for other purposes.
