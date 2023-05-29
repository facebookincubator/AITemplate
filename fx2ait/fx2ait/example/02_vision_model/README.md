# ResNet-18

In this example, we will demo how to use FX2AIT for inference on the ResNet-18 model from torchvision.

## Code structure
```
test_vision_model.py            # ResNet definition using torch API
../benchmark_utils.py           # Accuracy verification and Benchmark code for FX2AIT
```

## How to Use
FX2AIT allows users to directly define a torch model, while fx2ait converter does the conversion for the usage.
Therefore the definition of model is as simple as
```
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torchvision.models.resnet18()

            def forward(self, x):
                return self.mod(x)
```
Notice that because AIT supports channel last, while pytorch supports channel first operation, FX2AIT automatically performs this layout conversion for you.

To run the test and benchmark,
```
python fx2ait/fx2ait/example/02_vision_model/test_vision_model.py
```

## Reference Speed vs PyTorch Eager

### A100-40GB / CUDA 11.6.2
_PT = PyTorch 1.12 Eager_

| Batch size | PT Latency (s) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) | Speedup    |
|------------|----------------|---------------|------------------|----------------|------------|
|          1 |     0.00349264 |    286.316562 |       0.00052888 |     1890.78465 | 6.60382564 |
|          8 |     0.00382057 |    2093.93053 |        0.0007766 |     10301.2714 | 4.91958606 |
|         16 |     0.00351062 |    4557.59936 |       0.00098235 |     16287.4093 | 3.57368167 |
|         32 |     0.00321071 |    9966.64244 |       0.00166504 |     19218.8053 | 1.92831291 |
|        256 |     0.01670636 |    15323.5057 |       0.01181243 |     21672.0808 | 1.41430305 |
|        512 |     0.03276252 |    15627.6137 |       0.02347752 |     21808.0915 | 1.39548442 |



### Note for Performance Results

- For NVIDIA A100, our test cluster doesn't allow us to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- Performance results are what we can reproduce and for reference only. It should not be used for other purposes.
