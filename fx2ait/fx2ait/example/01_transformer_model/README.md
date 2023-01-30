# Transfomer encoder

In this example, we will demo how to use FX2AIT for inference on the transformer encoder block from pytorch.

## Code structure
```
test_transformer_encoder.py     # Transformer encoder block definition using torch API
../benchmark_utils.py           # Accuracy verification and Benchmark code for FX2AIT
```

## How to Use
FX2AIT allows users to directly define a torch model, while fx2ait converter does the conversion for the usage.
Therefore the encoder can be defined normally as
```
        class EncoderBlock(torch.nn.Module):
            def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
                super().__init__()
                # Attention layer
                self.attn = torch.nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                # # Two-layer MLP
                self.linear_net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, dim_feedforward),
                    torch.nn.Dropout(dropout),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(dim_feedforward, input_dim),
                )
                # Layers to apply in between the main layers
                self.norm1 = torch.nn.LayerNorm(input_dim)
                self.norm2 = torch.nn.LayerNorm(input_dim)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x):
                # Attention part
                attn_out, _ = self.attn(query=x, key=x, value=x)
                # return attn_out
                x = x + self.dropout(attn_out)
                x = self.norm1(x)

                # MLP part
                linear_out = self.linear_net(x)
                x = x + self.dropout(linear_out)
                x = self.norm2(x)

                return x
```
To run the test and benchmark,
```
python fx2ait/fx2ait/example/01_transformer_model/test_transformer_encoder.py
```

## Reference Speed vs PyTorch Eager

### A100-40GB / CUDA 11.6.2
_PT = PyTorch 1.12 Eager_

| Batch size | PT Latency (s) | PT QPS (im/s) | AIT Latency (ms) | AIT QPS (im/s) | Speedup    |
|------------|----------------|---------------|------------------|----------------|------------|
|          1 |     0.00043845 |    2280.75893 |        0.0001806 |     5537.09872 | 2.42774396 |
|          8 |     0.00047376 |    8443.01343 |        0.0002221 |     18009.9959 | 2.13312416 |
|         16 |     0.00085377 |    18740.4255 |       0.00050193 |     31876.7364 | 1.90096119 |
|         32 |     0.00150154 |    21311.3919 |       0.00069578 |     45991.3908 | 2.15806602 |
|         64 |     0.00296888 |    21556.9773 |       0.00138113 |     46338.7065 | 2.14959202 |
|        128 |     0.00530519 |    24127.3232 |       0.00261813 |     48889.8245 | 2.02632609 |
|        256 |     0.01015745 |    25203.1791 |       0.00516545 |     49560.0242 | 1.96641955 |
|        512 |     0.02023099 |    25307.7086 |       0.01034528 |     49491.1828 | 1.95557739 |



### Note for Performance Results

- For NVIDIA A100, our test cluster doesn't allow us to lock frequency. We make warm up longer to collect more stable results, but it is expected to have small variance to the results with locked frequency.
- Performance results are what we can reproduce and for reference only. It should not be used for other purposes.
