#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Fuse decomposed bmm -> scale -> bias -> activation -> scale -> bmm pattern
into a single classic_b2b_bmm op.

Matches the pattern:
    score = bmm_rcr(Q, K)           # Q[B,M,K] @ K[B,N,K]^T -> [B,M,N]
    score = score * alpha0           # elementwise MUL with constant scalar
    score = score + bias             # elementwise ADD with bias tensor
    score = activation(score)        # elementwise SIGMOID / RELU / TANH / SILU / GELU
    score = score * alpha1           # elementwise MUL with constant scalar (optional)
    output = bmm_rrr(score, V)       # score[B,M,N] @ V[B,N,N1] -> [B,M,N1]

And replaces with:
    output = classic_b2b_bmm(Q, K, V, bias)

Note: classic_b2b_bmm internally treats K as column-major [B, N0, K0], which
is exactly what bmm_rcr provides. The second GEMM (score @ V) is row-major
on both sides, matching bmm_rrr.
"""

import logging
from typing import List, Optional, Tuple

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType
from aitemplate.compiler.ops.b2b_bmm.classic_b2b_bmm import classic_b2b_bmm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_utils import (
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    replace_tensor,
    sanitize_sorted_graph,
)


_LOGGER = logging.getLogger(__name__)

# Map from AIT FuncEnum to CUTLASS EpilogueMathName string
_FUNC_TO_EPILOGUE = {
    FuncEnum.SIGMOID: "Sigmoid",
    FuncEnum.RELU: "ReLu",
    FuncEnum.TANH: "Tanh",
    FuncEnum.SILU: "SiLu",
    FuncEnum.GELU: "Gelu",
}


def _extract_only_one_op(ops) -> Optional[Operator]:
    if ops is None or len(ops) != 1:
        return None
    return list(ops)[0]


def _get_single_dst_op(tensor: Tensor) -> Optional[Operator]:
    """Return the single downstream op of a tensor, or None."""
    dst_ops = tensor._attrs.get("dst_ops")
    if dst_ops is None or len(dst_ops) != 1:
        return None
    return list(dst_ops)[0]


def _is_bmm_rcr(op: Operator) -> bool:
    return op._attrs["op"] == "bmm_rcr"


def _is_bmm_rrr(op: Operator) -> bool:
    return op._attrs["op"] == "bmm_rrr"


def _is_first_bmm(op: Operator) -> bool:
    """Check if op is a valid first BMM (Q @ K^T).
    Supports bmm_rcr (K stored as [B,N,K] column-major)."""
    return _is_bmm_rcr(op)


def _is_elementwise(op: Operator, func: FuncEnum) -> bool:
    return op._attrs["op"] == "elementwise" and op._attrs["func"] == func


def _is_activation(op: Operator) -> bool:
    """Check if op is a supported activation elementwise op."""
    if op._attrs["op"] != "elementwise":
        return False
    return op._attrs["func"] in _FUNC_TO_EPILOGUE


def _get_const_scalar(tensor: Tensor) -> Optional[float]:
    """If tensor is a constant scalar, return its float value."""
    if tensor.is_a_const_num():
        val = tensor._attrs["value"]
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _get_mul_const_and_tensor(
    op: Operator, upstream_tensor: Tensor
) -> Optional[Tuple[float, Tensor]]:
    """For an elementwise MUL op, extract (const_value, other_tensor).

    Returns None if the op is not a MUL with one constant scalar operand,
    or if the upstream_tensor is not one of the operands.
    """
    if not _is_elementwise(op, FuncEnum.MUL):
        return None
    args = op._attrs["args"]
    if len(args) != 2:
        return None

    if args[0] is upstream_tensor:
        const_val = _get_const_scalar(args[1])
        if const_val is not None:
            return (const_val, args[1])
    elif args[1] is upstream_tensor:
        const_val = _get_const_scalar(args[0])
        if const_val is not None:
            return (const_val, args[0])
    return None


def _get_add_bias_tensor(
    op: Operator, upstream_tensor: Tensor
) -> Optional[Tensor]:
    """For an elementwise ADD op, extract the bias tensor (the non-upstream operand).

    Returns None if the op is not an ADD or if neither operand is the upstream tensor.
    """
    if not _is_elementwise(op, FuncEnum.ADD):
        return None
    args = op._attrs["args"]
    if len(args) != 2:
        return None

    if args[0] is upstream_tensor:
        return args[1]
    elif args[1] is upstream_tensor:
        return args[0]
    return None


def _try_match_b2b_bmm(bmm0_output: Tensor) -> Optional[dict]:
    """Try to match the b2b_bmm pattern starting from the output of the first bmm.

    Pattern:
        bmm0_out = bmm_rcr(Q, K)       # Q[B,M,K] @ K[B,N,K]^T -> [B,M,N]
        scaled = elementwise(MUL)(bmm0_out, alpha0)
        biased = elementwise(ADD)(scaled, bias)
        activated = elementwise(ACTIVATION)(biased)
        [optionally: alpha1_scaled = elementwise(MUL)(activated, alpha1)]
        output = bmm_rrr(alpha1_scaled_or_activated, V)  # score @ V

    Returns a dict with all extracted info, or None if pattern doesn't match.
    """
    # Step 0: Verify bmm0_out comes from a single bmm_rcr
    bmm0_op = _extract_only_one_op(bmm0_output._attrs["src_ops"])
    if bmm0_op is None or not _is_first_bmm(bmm0_op):
        return None

    # bmm0_out must not be a graph output and must have exactly one consumer
    if bmm0_output._attrs.get("is_output", False):
        return None

    Q, K = bmm0_op._attrs["inputs"][0], bmm0_op._attrs["inputs"][1]

    # Step 1: MUL with alpha0
    mul0_op = _get_single_dst_op(bmm0_output)
    if mul0_op is None:
        return None
    mul0_result = _get_mul_const_and_tensor(mul0_op, bmm0_output)
    if mul0_result is None:
        return None
    alpha0, alpha0_const_tensor = mul0_result
    mul0_out = mul0_op._attrs["outputs"][0]
    if mul0_out._attrs.get("is_output", False):
        return None

    # Step 2: ADD with bias
    add_op = _get_single_dst_op(mul0_out)
    if add_op is None:
        return None
    bias_tensor = _get_add_bias_tensor(add_op, mul0_out)
    if bias_tensor is None:
        return None
    add_out = add_op._attrs["outputs"][0]
    if add_out._attrs.get("is_output", False):
        return None

    # Step 3: Activation
    act_op = _get_single_dst_op(add_out)
    if act_op is None or not _is_activation(act_op):
        return None
    activation_func = act_op._attrs["func"]
    epilogue_math_name = _FUNC_TO_EPILOGUE[activation_func]
    act_out = act_op._attrs["outputs"][0]
    if act_out._attrs.get("is_output", False):
        return None

    # Step 4: Optional MUL with alpha1, then bmm_rrr
    next_op = _get_single_dst_op(act_out)
    if next_op is None:
        return None

    alpha1 = 1.0
    alpha1_const_tensor = None
    alpha1_divide_by_seq_len = False
    bmm1_input_tensor = act_out

    # Collect all intermediate tensors/ops to remove
    intermediate_tensors = [bmm0_output, mul0_out, add_out, act_out]
    intermediate_ops = [bmm0_op, mul0_op, add_op, act_op]

    if _is_elementwise(next_op, FuncEnum.MUL):
        # Optional alpha1 scaling
        mul1_result = _get_mul_const_and_tensor(next_op, act_out)
        if mul1_result is not None:
            alpha1, alpha1_const_tensor = mul1_result
            mul1_out = next_op._attrs["outputs"][0]
            if mul1_out._attrs.get("is_output", False):
                return None
            bmm1_input_tensor = mul1_out
            intermediate_tensors.append(mul1_out)
            intermediate_ops.append(next_op)

            # Now look for bmm_rrr after the alpha1 MUL
            next_op = _get_single_dst_op(mul1_out)
            if next_op is None:
                return None

    # Step 5: Final bmm_rrr(score, V)
    if not _is_bmm_rrr(next_op):
        return None
    bmm1_op = next_op
    bmm1_inputs = bmm1_op._attrs["inputs"]
    if len(bmm1_inputs) != 2:
        return None

    # The score tensor must be one of the inputs
    if bmm1_inputs[0] is bmm1_input_tensor:
        V = bmm1_inputs[1]
    elif bmm1_inputs[1] is bmm1_input_tensor:
        V = bmm1_inputs[0]
    else:
        return None

    bmm1_out = bmm1_op._attrs["outputs"][0]
    intermediate_ops.append(bmm1_op)

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "bias": bias_tensor,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha1_divide_by_seq_len": alpha1_divide_by_seq_len,
        "epilogue_math_name": epilogue_math_name,
        "bmm1_out": bmm1_out,
        "intermediate_tensors": intermediate_tensors,
        "intermediate_ops": intermediate_ops,
        "alpha0_const_tensor": alpha0_const_tensor,
        "alpha1_const_tensor": alpha1_const_tensor,
    }


def fuse_b2b_bmm(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Fuse decomposed attention pattern into classic_b2b_bmm.

    Matches: bmm_rcr -> MUL(alpha0) -> ADD(bias) -> activation -> [MUL(alpha1)] -> bmm_rrr
    Replaces with: classic_b2b_bmm(Q, K, V, bias)

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input sorted graph
    workdir : str, optional
        Working directory

    Returns
    -------
    List[Tensor]
        Transformed graph with fused classic_b2b_bmm ops
    """
    to_remove = set()
    output_tensors = []
    has_modified = False

    for tensor in sorted_graph:
        if tensor in to_remove:
            continue

        if tensor._attrs.get("is_output", False):
            output_tensors.append(tensor)

        # Look for tensors produced by bmm_rcr as fusion start points
        src_op = _extract_only_one_op(tensor._attrs["src_ops"])
        if src_op is None or not _is_first_bmm(src_op):
            continue

        match = _try_match_b2b_bmm(tensor)
        if match is None:
            continue

        Q = match["Q"]
        K = match["K"]
        V = match["V"]
        bias = match["bias"]
        alpha0 = match["alpha0"]
        alpha1 = match["alpha1"]
        epilogue_math_name = match["epilogue_math_name"]
        bmm1_out = match["bmm1_out"]

        _LOGGER.info(
            "Fusing b2b_bmm pattern: alpha0=%.4f, alpha1=%.4f, "
            "activation=%s -> classic_b2b_bmm",
            alpha0,
            alpha1,
            epilogue_math_name,
        )

        # Create the fused classic_b2b_bmm op
        b2b_op = classic_b2b_bmm(
            causal_type=CausalType.NO_CAUSAL,
            epilogue_math_name=epilogue_math_name,
            alpha0=alpha0,
            alpha1=alpha1,
            alpha1_divide_by_seq_len=match["alpha1_divide_by_seq_len"],
        )

        # Build the fused op output
        new_tensor = b2b_op(Q, K, V, bias)
        copy_tensor_attributes(new_tensor, bmm1_out)
        replace_tensor(bmm1_out, new_tensor)

        if new_tensor._attrs.get("is_output", False):
            output_tensors.append(new_tensor)

        # Remove references from intermediate ops to their input tensors
        for op in match["intermediate_ops"]:
            remove_dst_op_from_tensor(list(op._attrs["inputs"]), op)

        # Mark intermediate tensors for removal
        for t in match["intermediate_tensors"]:
            to_remove.add(t)

        has_modified = True

    if has_modified:
        sorted_graph = toposort(output_tensors)
        sorted_graph = sanitize_sorted_graph(sorted_graph)

    return sorted_graph
