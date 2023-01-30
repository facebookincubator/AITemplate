from aitemplate.utils.torch_utils import torch_dtype_to_string


def dtype_to_str(dtype):
    if dtype is None:
        return "float16"
    return torch_dtype_to_string(dtype)


def make_str_ait_friendly(s: str) -> str:
    if s.isalnum():
        ret = s
    else:
        ret = "".join(c if c.isalnum() else "_" for c in s)
    if ret[0].isdigit():
        ret = "_" + ret
    return ret
