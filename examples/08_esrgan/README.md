## ESRGAN Example

In this example, we show how to build fast AIT modules for ESRGAN models, and benchmark/run them.

### Build Dependencies

First, clone, build, and install AITemplate [per the README instructions](https://github.com/facebookincubator/AITemplate#clone-the-code).

This AIT ESRGAN example depends on `torch`, `click` and optionally `safetensors`. You could install them using `pip`.

### Download the ESRGAN model

We have tested the following ESRGAN models.

[RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Model architecture:
```
num_in_ch: 3,
num_out_ch: 3,
num_feat: 64,
num_block: 23,
num_grow_ch: 32,
scale: 4,
```


[RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

Model architecture:
```
num_in_ch: 3,
num_out_ch: 3,
num_feat: 64,
num_block: 6,
num_grow_ch: 32,
scale: 4,
```


[RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)

Model architecture:
```
num_in_ch: 3,
num_out_ch: 3,
num_feat: 64,
num_block: 23,
num_grow_ch: 32,
scale: 2,
```

A database of ESRGAN models can be found [here](https://upscale.wiki/wiki/Model_Database).

Safetensors versions are supported.


### Build AIT modules for ESRGAN

Build the AIT modules by running `compile.py`.

```
Usage: compile.py [OPTIONS]

Options:
  --model-path TEXT               model path. supports torch or safetensors
  --width <INTEGER INTEGER>...    Minimum and maximum width
  --height <INTEGER INTEGER>...   Minimum and maximum height
  --batch-size <INTEGER INTEGER>...
                                  Minimum and maximum batch size
  --include-constants BOOLEAN     include constants (model weights) with
                                  compiled model
  --num-in-ch INTEGER             Number of in channels
  --num-out-ch INTEGER            Number of out channels
  --num-feat INTEGER              Number of intermediate features
  --num-block INTEGER             Number of RRDB layers
  --num-grow-ch INTEGER           Number of channels for each growth
  --scale INTEGER                 Scale
  --use-fp16-acc BOOLEAN          use fp16 accumulation
  --convert-conv-to-gemm BOOLEAN  convert 1x1 conv to gemm
  --work-dir TEXT                 Work directory
  --model-name TEXT               Model name
  --help                          Show this message and exit.
```

Use `--num-in-ch`, `--num-out-ch`, `--num-feat`, `--num-block`, `--num-grow-ch`, `--scale` options to set the ESRGAN model architecture. The default values are for `RealESRGAN_x4plus` architecture.

`--width` and `--height` require a minimum and maximum value, the compiled module supports the range of resolutions. However, with 2x model architecture, only static shape is supported, the maximum value for each dimension is used. Defaults are `64` and `1024`.

`--batch-size` is supported for 4x model architecture, provide minimum and maximum values. Default is `1 1`.

Use `--include-constants False` to compile the module without model weights.

AIT modules are compatible with all ESRGAN models with the same model architecture. This can simplify deployment by compiling a module without model weights then applying weights at runtime by using AIT mapped weights (see `map_rrdb` in `./modeling/rrdbnet.py`) with the module's `set_many_constants_with_tensors`.

In our tests an ESRGAN module compiled with weights is approximately `~38MB`, and `~6.5MB` without.

Examples:

```
python compile.py --model-path "RealESRGAN_x4plus.safetensors"
```

```
python compile.py --model-path "RealESRGAN_x4plus_anime_6B.pth" --num-block 6 --model-name RealESRGAN_x4plus_anime_6B
```

```
python compile.py --model-path "RealESRGAN_x2plus.pth" --scale 2 --model-name RealESRGAN_x2plus --width 512 512 --height 512 512
```


#### Multi-GPU profiling
AIT needs to do profiling to select the best algorithms for CUTLASS and CK.
To enable multiple GPUs for profiling, use the environment variable `CUDA_VISIBLE_DEVICES` on NVIDIA platform and `HIP_VISIBLE_DEVICES` on AMD platform.


### Run Models

`demo.py` provides example usage of ESRGAN modules.

```
Usage: demo.py [OPTIONS]

Options:
  --module-path TEXT        the AIT module path
  --input-image-path TEXT   path to input image
  --output-image-path TEXT  path to output image
  --scale INTEGER           Scale of ESRGAN model
  --help                    Show this message and exit.
```

`--scale` must match the scale of the model architecture.

Limitations:
* Demo does not support multiple images/batch size.
* Demo does not support images with alpha channel.
