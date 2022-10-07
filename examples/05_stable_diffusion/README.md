## Stable Diffusion Example

In this example, we show how to build fast AIT modules for CLIP, UNet, VAE models, and benchmark/run them.

### Build Dependencies

First, clone, build, and install AITemplate [per the README instructions](https://github.com/facebookincubator/AITemplate#clone-the-code).

This AIT stable diffusion example depends on `diffusers`, `transformers`, `torch` and `click`.

Verify the library versions. We have tested transformers 4.21/4.22/4.23, diffusers 0.3/0.4 and torch 1.11/1.12.

```
>>> import transformers
>>> transformers.__version__
'4.21.2'
>>> import diffusers
>>> diffusers.__version__
'0.3.0'
>>> torch.__version__
'1.12.1+cu116'
```

### Build AIT modules for CLIP, UNet, VAE

Build the AIT modules by running `compile.py`. You must first register in Hugging Face Hub to obtain an access token for the Stable Diffusion weights. See [user access tokens](https://huggingface.co/docs/hub/security-tokens) for more info. Your access tokens are listed in your [Hugging Face account settings](https://huggingface.co/settings/tokens).

```
python3 examples/05_stable_diffusion/compile.py --token ACCESS_TOKEN
```
It generates three folders: `./tmp/CLIPTextModel`, `./tmp/UNet2DConditionModel`, `./tmp/AutoencoderKL`. In each folder, there is a `test.so` file which is the generated AIT module for the model.

Compile the img2img models:
```
python3 examples/05_stable_diffusion/compile.py --img2img True --token ACCESS_TOKEN
```

#### Multi-GPU profiling
AIT needs to do profiling to select the best algorithms for CUTLASS and CK.
To enable multiple GPUs for profiling, use the environment variable `CUDA_VISIBLE_DEVICES` on NVIDIA platform and `HIP_VISIBLE_DEVICES` on AMD platform.

### Benchmark

This step is optional. You can run `benchmark.py` with the access token to initialize the weights and benchmark.

```
python3 examples/05_stable_diffusion/benchmark.py --token ACCESS_TOKEN
```

### Run Models

Run AIT models with an example image:

```
python3 examples/05_stable_diffusion/demo.py --token ACCESS_TOKEN
```

Img2img demo:

```
python3 examples/05_stable_diffusion/demo_img2img.py --token ACCESS_TOKEN
```

Check the resulted image: `example_ait.png`


### Sample outputs

Command: `python3 examples/05_stable_diffusion/demo.py --token hf_xxx --prompt "Mountain Rainier in van Gogh's world"`

![sample](https://raw.githubusercontent.com/AITemplate/webdata/main/imgs/example_ait_rainier.png)

Command: `python3 examples/05_stable_diffusion/demo.py --token hf_xxx --prompt "Sitting in a tea house in Japan with Mount Fuji in the background, sunset professional portrait, Nikon 85mm f/1.4G"`

![sample](https://raw.githubusercontent.com/AITemplate/webdata/main/imgs/example_ait_fuji.png)

Command: `python3 examples/05_stable_diffusion/demo.py --token hf_xxx --prompt "A lot of wild flowers with North Cascade Mountain in background, sunset professional photo, Unreal Engine"`

![sample](https://raw.githubusercontent.com/AITemplate/webdata/main/imgs/example_ait_cascade2.png)

## Results

_PT = PyTorch 1.12 Eager_

_OOM = Out of Memory_
### A100-40GB / CUDA 11.6, 50 steps

| Module   | PT Latency (ms) | AIT Latency (ms) |
|----------|-----------------|------------------|
| CLIP     | 9.48            | 0.87             |
| UNet     | 60.52           | 22.47            |
| VAE      | 47.78           | 37.43            |
| Pipeline | 3058.27         | 1282.98          |

- PT: 17.50 it/s
- AIT: 42.45 it/s

### RTX 3080-10GB / CUDA 11.6, 50 steps

| Module   | PT Latency (ms) | AIT Latency (ms) |
|----------|-----------------|------------------|
| CLIP     | OOM             | 0.85             |
| UNet     | OOM             | 40.22            |
| VAE      | OOM             | 44.12            |
| Pipeline | OOM             | 2163.43          |

- PT: OOM
- AIT: 24.51 it/s

### MI-250 1 GCD, 50 steps

| Module   | PT Latency (ms) | AIT Latency (ms) |
|----------|-----------------|------------------|
| CLIP     | 6.16            | 2.98             |
| UNet     | 78.42           | 62.18            |
| VAE      | 63.83           | 164.50           |
| Pipeline | 4300.16         | 3476.07          |

- PT: 12.43 it/s
- AIT: 15.60 it/s

## Batched Version

### A100-40GB / CUDA 11.6

- Stable Diffusion with AIT batch inference, 50 steps

| Batch size   | PT Latency (ms)  | AIT Latency (ms) |
|--------------|------------------|------------------|
|  1           |   3058.27        |      1282.98     |
|  3           |   7334.46        |      3121.88     |
|  8           |   17944.60       |      7492.81     |
|  16          |      OOM         |      14931.95    |

- AIT Faster rendering, 25 steps

| Batch size | AIT Latency (ms) | AVG im/s |
|------------|------------------|----------|
| 1          | 695              | 0.69     |
| 3          | 1651             | 0.55     |
| 8          | 3975             | 0.50     |
| 16         | 7906             | 0.49     |


## IMG2IMG

### A100-40GB / CUDA 11.6, 40 steps

| Module   | PT Latency (ms) | AIT Latency (ms) |
|----------|-----------------|------------------|
| Pipeline | 4163.60         | 1785.46          |



### Note for Performance Results

- For all benchmarks we render the images of size 512x512
- For img2img model we only support fix input 512x768 by default, stay tuned for dynamic shape support
- For NVIDIA A100, our test cluster doesn't allow to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- To benchmark MI-250 1 GCD, we lock the frequency with command `rocm-smi -d x --setperfdeterminism 1700`, where `x` is the GPU id.
- Performance results are what we can reproduced & take reference. It should not be used for other purposes.
