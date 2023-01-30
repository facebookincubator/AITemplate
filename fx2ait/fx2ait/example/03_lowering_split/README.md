# AIT Lowerer
Now let's go back the example of encoder. Imagine the cases that
1) You want to test the effect of a particular op, say MultiheadAttention on the entire module.
2) There is some special op that AIT doesn't support.
AIT actually provide an **automatic** Lowerer to split the graph into subgraphs and run interpreter,
so that AIT only run the part it can handle and leave other to AITemplate.

In this example, we will demo how to use AitLowerer for inference on any models.

## Code structure
```
test_lowerr.py                  # Splited transformer encoder block to illustrate the usage of AitLowerer.
../benchmark_utils.py           # Accuracy verification and Benchmark code for FX2AIT
../lower/
        lower.py                # Lower interface, which integrates lowering passes of Split subgraph and AIT Interpreter
        ait_splitter.py         # Splitter to split graph into submodules
        ait_setting.py          # Lowering settings

```

## How to Use
To skip an operation can be extremely easy. One just need to register in the method function `@torch.fx.wrap`
```
@torch.fx.wrap
def unsupported_attention_op(f, x):
    attn_out, _ = f(x, x, x)
    return attn_out
```
Then at forward stage, call the function.
```
        class LowerModule(torch.nn.Module):
            def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                ...

            def forward(self, x):
                # Unsupported op will not be lowered to AIT backend.
                attn_out = unsupported_attention_op(self.attn, x)
                ...
```
Then AIT won't deal with that part.
```
        lowerer = AitLowerer.create(
            LowerSettings(
                workdir="/tmp",
                name="test_ait_lower",
                min_acc_module_size=0,
            )
        )
        lowered = lowerer(model, inputs)
        lower_output = lowered(*inputs)
```
The mechanism is that Acc tracer allows user to register wrap function so that Acc won't deal with it.
Then our splitter will split the them into subgraph: _run_on_gpu_0 for pytorch Eager mode and _run_on_acc_1 for AIT,
where _run_on_gpu_0 contains torch.nn.MultiheadAttention and _run_on_acc_1 contains the rest of the model.
Finally, interpreter will be called for the AIT subgraph. (_run_on_acc_1)

*Notice that our splitter only split subgraphs with more than 10 ops, since otherwise the subgraph is too small.*

To run the test and benchmark,
```
python fx2ait/fx2ait/example/03_lowering_split/test_lower.py
```

## Reference Speed vs PyTorch Eager

### A100-40GB / CUDA 11.6.2
_PT = PyTorch 1.12 Eager_

| Batch size | PT Latency (s) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) | Speedup    |
|------------|----------------|---------------|------------------|----------------|------------|
| 1          | 0.00065761     | 1520.66428    | 0.00076476       | 1307.59981     | 0.85988724 |
| 4          | 0.00090687     | 4410.77681    | 0.00079056       | 5059.68597     | 1.14711902 |
| 16         | 0.00249116     | 6422.69897    | 0.00200686       | 7972.66574     | 1.24132639 |
| 32         | 0.00473638     | 6756.209      | 0.00396992       | 8060.62008     | 1.19306849 |
| 64         | 0.00914742     | 6996.51201    | 0.00754977       | 8477.07749     | 1.2116148  |
| 128        | 0.0178672      | 7163.96537    | 0.01501702       | 8523.66305     | 1.1897968  |
| 256        | 0.03554306     | 7202.53192    | 0.02998132       | 8538.65123     | 1.18550689 |
| 512        | 0.07118476     | 7192.55069    | 0.06006168       | 8524.56943     | 1.18519421 |

From the example, we learn without AIT's multihead attention module, the speedup will be degraded to 1.2x compared to Pytorch Eager.

### Note for Performance Results
- For NVIDIA A100, our test cluster doesn't allow us to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- Performance results are what we can reproduce and for reference only. It should not be used for other purposes.
