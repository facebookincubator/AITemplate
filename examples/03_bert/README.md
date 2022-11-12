# BERT

This directory contains an AIT demo for the [BERT language representation model](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/bert).

Only `bert-base-uncased` is included.

## Prerequisites

Install the dependencies:
```
python3 -m pip install transformers click torch
```

## Benchmarking

To run a basic benchmark, use `benchmark.py`:

```
python3 examples/03_bert/benchmark_ait.py
```

There are two options for hidden activations, `gelu` and `fast_gelu` (`fast_gelu` by default).
`gelu` is not supported on AMD hardware yet.

```
python3 examples/03_bert/benchmark_ait.py --activation gelu
python3 examples/03_bert/benchmark_ait.py --activation fast_gelu
```

The batch size and sequence length can also be configured via the command line:
```
python3 examples/03_bert/benchmark_ait.py --batch-size 1 --seq-length 128
```

PyTorch eager mode benchmarks can also be run:
```
python3 examples/03_bert/benchmark_pt.py
```

To benchmark BERT embeddings, run benchmark with `--encoders-only False`

## Quick Demo

To run a quick demo with a simple prompt, use `demo.py`:
```
python3 examples/03_bert/demo.py --prompt "The quick brown fox jumps over the lazy dog."
```

The demo prints out the resulting logits. The demo only works with sequence length <= 512.

## Multi-GPU profiling
AIT requires to do profiling to decide best algorithms for CUTLASS and CK.
To enable multiple GPUs profiling, use the environment variable `CUDA_VISIBLE_DEVICES` on NVIDIA platform and `HIP_VISIBLE_DEVICES` on AMD platform.

## Reference Speed vs PyTorch Eager
_PT = PyTorch 1.12 Eager_
_OOM = Out of Memory_

### A100-40GB / CUDA 11.6.2

- Sequence length 64

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 7.96            | 125.65         | 0.71             | 1399.64         |
| 2          | 8.38            | 238.59         | 0.74             | 2719.15         |
| 4          | 8.29            | 482.30         | 0.80             | 4994.37         |
| 8          | 8.51            | 939.97         | 0.95             | 8439.67         |
| 16         | 8.09            | 1978.47        | 1.41             | 11385.85        |
| 32         | 9.19            | 3481.34        | 2.23             | 14357.58        |
| 64         | 9.12            | 7016.80        | 4.14             | 15458.15        |
| 128        | 14.52           | 8814.57        | 8.00             | 15991.44        |
| 256        | 27.75           | 9224.39        | 15.99            | 16006.79        |


- Sequence length 128

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 8.02            | 124.72         | 0.78             | 1281.52         |
| 2          | 8.29            | 241.22         | 0.85             | 2364.94         |
| 4          | 8.51            | 470.29         | 0.99             | 4044.33         |
| 8          | 8.12            | 985.72         | 1.43             | 5600.93         |
| 16         | 9.22            | 1735.20        | 2.21             | 7232.47         |
| 32         | 9.11            | 3512.80        | 4.17             | 7677.82         |
| 64         | 15.29           | 4184.93        | 8.05             | 7949.06         |
| 128        | 29.44           | 4347.33        | 16.03            | 7987.11         |
| 256        | 56.34           | 4543.88        | 31.57            | 8109.08         |


- Sequence length 384

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 8.72            | 114.73         | 1.63             | 611.91          |
| 2          | 8.31            | 240.73         | 1.97             | 1013.19         |
| 4          | 8.64            | 463.10         | 2.55             | 1569.23         |
| 8          | 9.32            | 858.70         | 3.95             | 2025.62         |
| 16         | 13.90           | 1151.03        | 6.80             | 2354.21         |
| 32         | 26.72           | 1197.74        | 13.30            | 2405.46         |
| 64         | 51.02           | 1254.34        | 26.68            | 2398.95         |
| 128        | 100.26          | 1276.67        | 51.60            | 2480.67         |
| 256        | OOM             | OOM            | 101.55           | 2520.81         |


- Sequence length 1024

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 9.74            | 102.65         | 2.20             | 454.12          |
| 2          | 11.38           | 175.75         | 4.15             | 481.95          |
| 4          | 13.61           | 293.90         | 8.36             | 478.44          |
| 8          | 25.79           | 310.15         | 12.53            | 638.53          |
| 16         | 49.91           | 320.59         | 21.61            | 740.48          |
| 32         | 97.00           | 329.91         | 42.84            | 746.88          |
| 64         | 191.14          | 334.83         | 83.95            | 762.39          |
| 128        | OOM             | OOM            | 163.96           | 780.70          |
| 256        | OOM             | OOM            | 324.22           | 789.58          |



- Sequence length 4096

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 32.82           | 30.47          | 18.23            | 54.87           |
| 2          | 65.25           | 30.65          | 35.64            | 56.11           |
| 4          | 128.73          | 31.07          | 103.67           | 38.58           |
| 8          | OOM             | OOM            | 119.45           | 66.98           |
| 16         | OOM             | OOM            | 166.25           | 96.24           |
| 32         | OOM             | OOM            | 333.98           | 95.81           |
| 64         | OOM             | OOM            | 662.29           | 96.63           |
| 128        | OOM             | OOM            | 1313.77          | 97.43           |
| 256        |                 |                |                  |                 |



### MI-250 / ROCm 5.2.3 / HIPCC-10736

#### 1 GCD

- Sequence length 64

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 5.72            | 174.72         | 2.78             | 359.88          |
| 2          | 5.96            | 335.38         | 2.87             | 697.76          |
| 4          | 5.85            | 684.16         | 2.85             | 1404.31         |
| 8          | 6.15            | 1300.72        | 3.15             | 2540.72         |
| 16         | 6.14            | 2605.40        | 3.78             | 4231.12         |
| 32         | 7.73            | 4138.06        | 5.34             | 5993.50         |
| 64         | 14.38           | 4451.07        | 9.10             | 7030.42         |
| 128        | 26.18           | 4889.95        | 16.45            | 7780.40         |
| 256        | 49.95           | 5125.04        | 31.90            | 8023.98         |


- Sequence length 128

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 5.76            | 173.55         | 2.68             | 373.03          |
| 2          | 6.06            | 330.18         | 2.87             | 697.33          |
| 4          | 5.96            | 670.65         | 3.02             | 1324.91         |
| 8          | 6.03            | 1326.23        | 3.65             | 2194.62         |
| 16         | 9.35            | 1711.55        | 4.98             | 3212.12         |
| 32         | 16.46           | 1943.61        | 8.48             | 3775.22         |
| 64         | 30.83           | 2075.74        | 15.44            | 4146.40         |
| 128        | 58.74           | 2179.24        | 30.57            | 4187.68         |
| 256        | 115.27          | 2220.87        | 59.28            | 4318.61         |


- Sequence length 384

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 5.78            | 172.87         | 2.97             | 336.14          |
| 2          | 6.02            | 332.30         | 3.45             | 579.89          |
| 4          | 8.00            | 499.85         | 4.68             | 854.16          |
| 8          | 13.79           | 580.01         | 7.47             | 1070.24         |
| 16         | 24.39           | 656.06         | 13.04            | 1226.77         |
| 32         | 45.56           | 702.33         | 24.26            | 1318.80         |
| 64         | 87.84           | 728.57         | 47.87            | 1336.92         |
| 128        | 172.57          | 741.71         | 95.22            | 1344.26         |
| 256        | 352.27          | 726.71         | 185.94           | 1376.78         |



- Sequence length 1024

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 6.86            | 145.71         | 4.20             | 237.84          |
| 2          | 12.41           | 161.21         | 5.82             | 343.62          |
| 4          | 22.25           | 179.80         | 10.20            | 392.26          |
| 8          | 41.94           | 190.73         | 18.91            | 423.05          |
| 16         | 81.03           | 197.45         | 37.86            | 422.60          |
| 32         | 159.06          | 201.19         | 71.65            | 446.62          |
| 64         | 321.51          | 199.06         | 148.86           | 429.95          |
| 128        | OOM             | OOM            | 277.53           | 461.21          |
| 256        | OOM             | OOM            | 563.07           | 454.65          |


- Sequence length 4096

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          | 49.89           | 20.04          | 16.18            | 61.81           |
| 2          | 93.22           | 21.45          | 30.67            | 65.21           |
| 4          | 183.57          | 21.79          | 66.78            | 59.90           |
| 8          | 366.57          | 21.82          | 117.49           | 68.09           |
| 16         | OOM             | OOM            | 231.15           | 69.22           |
| 32         | OOM             | OOM            | 459.46           | 69.65           |
| 64         | OOM             | OOM            | 1031.86          | 62.02           |
| 128        |                 |                |                  |                 |
| 256        |                 |                |                  |                 |


#### 2 GCDs

- Sequence length 64

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          |                 |                |                  |                 |
| 2          | 5.52            | 362.55         | 2.80             | 714.99          |
| 4          | 6.04            | 661.73         | 2.89             | 1385.05         |
| 8          | 6.07            | 1317.20        | 2.82             | 2835.38         |
| 16         | 6.02            | 2659.82        | 3.29             | 4866.99         |
| 32         | 6.09            | 5257.45        | 3.83             | 8352.10         |
| 64         | 8.53            | 7506.95        | 5.81             | 11013.02        |
| 128        | 15.34           | 8346.14        | 10.00            | 12806.23        |
| 256        | 28.44           | 9002.30        | 18.92            | 13528.13        |


- Sequence length 128

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          |                 |                |                  |                 |
| 2          | 5.58            | 358.62         | 2.68             | 745.20          |
| 4          | 6.20            | 644.91         | 2.83             | 1411.55         |
| 8          | 6.08            | 1316.09        | 3.21             | 2492.88         |
| 16         | 5.89            | 2716.79        | 3.86             | 4144.50         |
| 32         | 9.86            | 3247.03        | 5.41             | 5915.33         |
| 64         | 17.71           | 3614.25        | 9.64             | 6640.53         |
| 128        | 32.74           | 3909.15        | 17.81            | 7186.25         |
| 256        | 62.73           | 4080.77        | 35.73            | 7165.20         |


- Sequence length 384

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          |                 |                |                  |                 |
| 2          | 5.57            | 358.88         | 3.09             | 647.71          |
| 4          | 6.12            | 653.83         | 3.62             | 1104.69         |
| 8          | 8.35            | 958.19         | 4.94             | 1620.06         |
| 16         | 14.29           | 1119.38        | 8.29             | 1930.01         |
| 32         | 26.10           | 1226.17        | 14.96            | 2139.07         |
| 64         | 50.01           | 1279.72        | 28.22            | 2268.02         |
| 128        | 97.55           | 1312.15        | 55.94            | 2288.37         |
| 256        | 193.00          | 1326.44        | 111.27           | 2300.68         |



- Sequence length 1024

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          |                 |                |                  |                 |
| 2          | 6.80            | 294.16         | 4.36             | 458.93          |
| 4          | 13.01           | 307.55         | 6.43             | 622.23          |
| 8          | 23.39           | 341.99         | 11.52            | 694.52          |
| 16         | 44.45           | 359.94         | 21.83            | 732.90          |
| 32         | 87.23           | 366.84         | 43.73            | 731.77          |
| 64         | 172.92          | 370.12         | 82.92            | 771.85          |
| 128        | 352.09          | 363.54         | 173.14           | 739.29          |
| 256        | OOM             | OOM            | 322.97           | 792.64          |


- Sequence length 4096

| Batch size | PT Latency (ms) | PT QPS (seq/s) | AIT Latency (ms) | AIT QPS (seq/s) |
|------------|-----------------|----------------|------------------|-----------------|
| 1          |                 |                |                  |                 |
| 2          | 54.67           | 36.58          | 18.31            | 109.23          |
| 4          | 104.19          | 38.39          | 35.09            | 113.99          |
| 8          | 206.62          | 38.72          | 77.03            | 103.86          |
| 16         | 412.58          | 38.78          | 133.59           | 119.77          |
| 32         | OOM             | OOM            | 263.40           | 121.49          |
| 64         | OOM             | OOM            | 524.11           | 122.11          |
| 128        | OOM             | OOM            | 1186.20          | 107.91          |
| 256        |                 |                |                  |                 |


### Note Performance Results

- For NVIDIA A100, our test cluster doesn't allow to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- To benchmark MI-250, the first step is to run `python3 benchmark_ait.py` to generate all necessary model dynamic library files with single GCD. Then run `./benchmark_mi250.sh {batch_size}` to simulate data parallel execution on 2 GCDs, each GCD is processing half of the batch.
- To benchmark MI-250 1 GCD, we lock the frequency with command `rocm-smi -d x --setperfdeterminism 1700`, where `x` is the GPU id.
- To benchmark MI-250 2 GCDs, we observed performance regression with rocm perf-determ mode. The 2 GCDs number is running without perf-determ mode set with command `rocm-smi -d x --resetperfdeterminism`, where `x` is the GPU id.
- PyTorch Eager result doesn't reflect [BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/), mainly due to BetterTransformer integration to TIMM/Transformer package is not yet landed.
- Performance results are what we can reproduced. It should not be used for other purposes.
