# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Functions for working with torch Tensors.
AITemplate doesn't depend on PyTorch, but it exposes
many APIs that work with torch Tensors anyways.

The functions in this file may assume that
`import torch` will work.
"""


def torch_dtype_to_string(dtype):
    import torch

    dtype_to_str = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    if dtype not in dtype_to_str:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(dtype_to_str.keys())}"
        )
    return dtype_to_str[dtype]
