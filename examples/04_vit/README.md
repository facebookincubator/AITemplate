# Vision Transformer (VIT)

In this example, we will demo how to lower a pretrained Vision Transformer from TIMM, and run inference in AITemplate. We tested on two variants of Vision Transformer: Base version with 224x224 input / patch 16, and Large version with 384x384 input / patch 16.

## Code structure
```
modeling
    vision_transformer.py    # VIT definition using AIT's frontend API
weight_utils.py              # Utils to convert TIMM VIT weights to AIT
verification.py              # Numerical verification between TIMM and AIT
benchmark_pt.py              # Benchmark code for PyTorch
benchmark_ait.py             # Benchmark code for AITemplate
```

## Reference Speed vs PyTorch Eager

### A100-40GB / CUDA 11.6.2
_PT = PyTorch 1.12 Eager_

- vit_base_patch16_224

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 4.95            | 202.15        | 1.02             | 979.31         |
| 2          | 5.26            | 380.43        | 1.15             | 1735.64        |
| 4          | 5.51            | 726.08        | 1.57             | 2543.72        |
| 8          | 5.56            | 1439.03       | 2.20             | 3642.16        |
| 16         | 8.59            | 1863.35       | 3.64             | 4396.74        |
| 32         | 15.95           | 2006.62       | 6.51             | 4916.93        |
| 64         | 31.48           | 2032.77       | 12.67            | 5052.52        |
| 128        | 59.86           | 2138.35       | 25.10            | 5099.77        |
| 256        | 115.00          | 2226.10       | 48.55            | 5273.03        |


- vit_large_patch16_384

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 9.88            | 101.17        | 3.84             | 260.21         |
| 2          | 11.90           | 168.02        | 5.87             | 340.98         |
| 4          | 21.20           | 188.66        | 11.49            | 348.09         |
| 8          | 39.33           | 203.43        | 19.09            | 419.07         |
| 16         | 76.00           | 210.54        | 36.19            | 442.08         |
| 32         | 147.24          | 217.33        | 70.03            | 456.93         |
| 64         | 291.00          | 219.93        | 135.25           | 473.21         |
| 128        | 578.99          | 221.08        | 267.09           | 479.24         |
| 256        | 1204.16         | 212.60        | 538.97           | 474.98         |


### MI-250 / ROCm 5.2.3 / HIPCC-10736
_PT = PyTorch 1.12 Eager_

#### 1 GCD

- vit_base_patch16_224

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 3.54            | 282.12        | 3.49             | 286.26         |
| 2          | 4.43            | 451.73        | 3.78             | 528.84         |
| 4          | 6.09            | 657.02        | 4.05             | 986.95         |
| 8          | 9.65            | 829.27        | 5.31             | 1507.06        |
| 16         | 16.62           | 962.98        | 8.50             | 1882.72        |
| 32         | 29.87           | 1071.25       | 14.43            | 2218.07        |
| 64         | 56.58           | 1131.08       | 26.52            | 2413.45        |
| 128        | 110.28          | 1160.73       | 51.62            | 2479.69        |
| 256        | 217.07          | 1179.35       | 102.82           | 2489.89        |



- vit_large_patch16_384

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          | 12.90           | 77.51         | 9.70             | 103.05         |
| 2          | 22.42           | 89.19         | 13.40            | 149.29         |
| 4          | 38.16           | 104.83        | 22.12            | 180.86         |
| 8          | 70.58           | 113.35        | 38.46            | 208.00         |
| 16         | 136.28          | 117.40        | 70.44            | 227.15         |
| 32         | 261.97          | 122.15        | 138.14           | 231.65         |
| 64         | 541.90          | 118.10        | 270.01           | 237.02         |
| 128        | 1108.36         | 115.49        | 534.97           | 239.27         |
| 256        | 2213.09         | 115.68        | 1063.24          | 240.77         |


#### 2 GCDs

- vit_base_patch16_224

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          |                 |               |                  |                |
| 2          | 3.49            | 572.95        | 3.59             | 556.55         |
| 4          | 4.11            | 974.26        | 3.97             | 1006.80        |
| 8          | 5.88            | 1359.64       | 4.23             | 1889.44        |
| 16         | 9.75            | 1641.06       | 5.71             | 2800.69        |
| 32         | 17.55           | 1823.03       | 9.34             | 3426.32        |
| 64         | 31.31           | 2043.79       | 16.24            | 3940.53        |
| 128        | 60.33           | 2121.64       | 30.97            | 4133.14        |
| 256        | 117.96          | 2170.29       | 59.82            | 4279.21        |


- vit_large_patch16_384

| Batch size | PT Latency (ms) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) |
|------------|-----------------|---------------|------------------|----------------|
| 1          |                 |               |                  |                |
| 2          | 12.73           | 157.07        | 10.52            | 190.13         |
| 4          | 22.97           | 174.12        | 14.94            | 267.82         |
| 8          | 39.78           | 201.08        | 24.55            | 325.85         |
| 16         | 74.95           | 213.48        | 43.95            | 364.07         |
| 32         | 146.18          | 218.91        | 82.04            | 390.06         |
| 64         | 283.04          | 226.12        | 162.62           | 393.55         |
| 128        | 583.03          | 219.54        | 313.34           | 408.51         |
| 256        | 1197.56         | 213.77        | 621.71           | 411.77         |



### Note for Performance Results

- For NVIDIA A100, our test cluster doesn't allow to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- To benchmark MI-250, the first step is to run `python3 benchmark_ait.py` to generate all necessary model dynamic library files with single GCD. Then run `./benchmark_mi250.sh {batch_size}` to simulate data parallel execution on 2 GCDs, each GCD is processing half of the batch.
- To benchmark MI-250 1 GCD, we lock the frequency with command `rocm-smi -d x --setperfdeterminism 1700`, where `x` is the GPU id.
- To benchmark MI-250 2 GCDs, we observed performance regression with rocm perf-determ mode. The 2 GCDs number is running without perf-determ mode set with command `rocm-smi -d x --resetperfdeterminism`, where `x` is the GPU id.
- PyTorch Eager result doesn't reflect [BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/), mainly due to BetterTransformer integration to TIMM/Transformer package is not yet landed.
- Performance results are what we can reproduce. It should not be used for other purposes.
