# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from aitemplate.compiler.base import Operator


def op_to_content(op: Operator):
    # TODO (XXX): Add op specialized attrs here, like gemm/conv
    content = {}
    content["op_type"] = op._attrs["op"]
    return content
