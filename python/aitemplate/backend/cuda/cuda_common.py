# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from typing import Dict

DTYPE_TO_CUDATYPE: Dict[str, str] = {
    "float16": "half",
    "float": "float",
    "int64": "int64_t",
}


DTYPE_TO_CUTLASSTYPE: Dict[str, str] = {
    "float16": "cutlass::half_t",
    "float": "float",
}


def dtype_to_cuda_type(dtype: str):
    """[summary]

    Parameters
    ----------
    dtype : str
        [description]

    Returns
    -------
    [type] str
        [description] return the corresponding cuda type
    """
    cuda_type = DTYPE_TO_CUDATYPE.get(dtype)

    if cuda_type is None:
        raise NotImplementedError("CUDA - Unsupported dtype: {}".format(dtype))
    return cuda_type


def dtype_to_cutlass_type(dtype: str):
    """[summary]

    Parameters
    ----------
    dtype : str
        [description]

    Returns
    -------
    [type] str
        [description] return the corresponding cuda type
    """
    cutlass_type = DTYPE_TO_CUTLASSTYPE.get(dtype)

    if cutlass_type is None:
        raise NotImplementedError("CUDA - Unsupported dtype: {}".format(dtype))
    return cutlass_type
