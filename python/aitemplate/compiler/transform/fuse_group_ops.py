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
Horizontal fusion pass to group ops together.
"""
import collections
import logging
import os
from typing import Callable, List, OrderedDict, Set

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_common import default_align_ab
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.fuse_split import _can_fuse_split_op
from aitemplate.compiler.transform.toposort import toposort

from aitemplate.utils import graph_utils
from aitemplate.utils.shape_utils import all_static_dimensions


_LOGGER = logging.getLogger(__name__)


# used by debugging only
def _dump_dependency_graph(graph, op_type, postfix, workdir):
    fname = f"fuse_group_{op_type}_dependency_graph_{postfix}.txt"
    file_path = os.path.join(workdir, fname)
    graph_str = []
    for parent, descendants in graph.items():
        parent_op = parent._attrs["name"]
        descendant_ops = [child._attrs["name"] for child in descendants]
        graph_str.append(f"[{parent_op}: {descendant_ops}];")

    with open(file_path, "w") as f:
        f.write("\n\n".join(graph_str))
        _LOGGER.info(f"Dumped dependency graph to {file_path}")


def _dump_groups(groups, op_type, workdir):
    fname = f"fuse_group_{op_type}_groups.txt"
    file_path = os.path.join(workdir, fname)
    with open(file_path, "w") as f:
        for group in groups:
            single_group_str = ",".join(op._attrs["name"] for op in group)
            f.write(f"[{single_group_str}]\n\n")
            f.write(graph_utils.sorted_op_pseudo_code(group))
            f.write("\n")
        _LOGGER.info(f"Dumped groups to {file_path}")


def _dump_single_group(group):
    single_group_str = ",".join(op._attrs["name"] for op in group)
    print(f"[{single_group_str}]\n")


def _ops_have_same_num_inputs(op1: Operator, op2: Operator) -> bool:
    """Helper function to check whether op1 and op2 can be grouped together"""
    num_inputs_1 = len(op1._attrs["inputs"])
    num_inputs_2 = len(op2._attrs["inputs"])
    return num_inputs_1 > 0 and num_inputs_1 == num_inputs_2


def _check_op_num_outputs(op: Operator, num_outputs: int) -> bool:
    return len(op._attrs["outputs"]) == num_outputs


def _get_ab_alignment(op: Operator) -> int:
    if op._attrs["op"].startswith("gemm_rcr"):
        k = op._attrs["inputs"][0]._size(1).value()
        return default_align_ab(k, k, op._attrs["inputs"][0].dtype())
    raise NotImplementedError(
        f"Need to add alignment check support for op {op._attrs['op']}"
    )


def _filter_gemm_op(op: Operator) -> bool:
    """Function to filter out bad gemm candidates for group_gemm."""
    if op._attrs["alpha"] != 1.0:
        return False

    # 2D input
    input = op._attrs["inputs"][0]
    if input._rank() != 2:
        return False

    # dynamic dim check
    if not all_static_dimensions(input.shape()):
        return False

    return _get_ab_alignment(op) > 1


def _gemm_op_check(op1: Operator, op2: Operator) -> bool:
    align1 = _get_ab_alignment(op1)
    align2 = _get_ab_alignment(op2)
    return align1 == align2


def _get_layernorm_flattened_normalized_shape(op: Operator) -> int:
    n = 1
    for shape in op._attrs["normalized_shape"]:
        n *= shape.value()
    return n


def _get_layernorm_alignment(n: int) -> int:
    if n % 4 == 0:
        return 4
    return 1


def _check_layernorm_n_match(op1: Operator, op2: Operator) -> int:
    """n is the flattened normalized_shape dim"""
    n1 = _get_layernorm_flattened_normalized_shape(op1)
    n2 = _get_layernorm_flattened_normalized_shape(op2)
    # we can't use half4 kernel here anyways
    if n1 < 128 and n2 < 128:
        return True

    align1 = _get_layernorm_alignment(n1)
    align2 = _get_layernorm_alignment(n2)
    if n1 <= 4096 and n2 <= 4096:
        if align1 != align2:
            return False

    # TODO: may need better heuristics
    # We may group 128, 256, 512, 1024 together if we only check neighbors, which might be ok.
    # We mostly want to rule out 128, 128, ..., 1024 cases
    if n1 >= 4 * n2 or 4 * n1 <= n2:
        return False

    if align1 == 4:

        def _in_range(n):
            return n >= 128 and n <= 4096

        # prefer to use half4 kernel
        n1_in_range = _in_range(n1)
        n2_in_range = _in_range(n2)
        return not (n1_in_range ^ n2_in_range)
    return True


def _layernorm_op_check(op1: Operator, op2: Operator) -> bool:
    """Function to filter out bad layernorm candidates for group_layernorm
    and group_layernorm_sigmoid_mul.
    """
    if op1 == op2:
        return True

    # check for same eps
    if op1._attrs["eps"] != op2._attrs["eps"]:
        return False

    if len(op1._attrs["inputs"]) != 3 or len(op2._attrs["inputs"]) != 3:
        return False

    # check for same rank and batch dims
    input1 = op1._attrs["inputs"][0]
    input2 = op2._attrs["inputs"][0]
    if input1._rank() != input2._rank():
        return False

    norm_shapes_1 = op1._attrs["normalized_shape"]
    norm_shapes_2 = op2._attrs["normalized_shape"]
    # This may be relaxed in the future
    if len(norm_shapes_1) != len(norm_shapes_2):
        return False

    # All batch dims must be the same
    for i in range(input1._rank() - len(norm_shapes_1)):
        if input1._size(i) != input2._size(i):
            return False

    # check gamma and bias
    if op1._attrs["gamma_constant"] is not None:
        return False

    if op1._attrs["beta_constant"] is not None:
        return False

    return _check_layernorm_n_match(op1, op2)


def _default_op_check(op1: Operator, op2: Operator = None) -> bool:
    return True


def _get_op_checker(op_type: str) -> Callable:
    """Returns op specific check functions to check if two ops can be
    grouped togher.
    """
    if op_type.startswith("gemm"):
        return _gemm_op_check
    if op_type.startswith("layernorm"):
        return _layernorm_op_check
    return _default_op_check


def _get_op_filter(op_type: str) -> Callable:
    """Returns op specific check functions to check for if a single op
    is good candidate for group ops.
    """
    if op_type.startswith("gemm"):
        return _filter_gemm_op
    return _default_op_check


_group_gemm_op_mapping = {
    "gemm_rcr": ops.group_gemm_rcr,
    "gemm_rcr_bias": ops.group_gemm_rcr_bias,
    "gemm_rcr_bias_relu": ops.group_gemm_rcr_bias_relu,
    "gemm_rcr_bias_sigmoid": ops.group_gemm_rcr_bias_sigmoid,
}


def _has_cycle(grouped_op: Operator, group: List[Operator]):
    """
    Assuming that grouped_op is in the group, determine if grouped_op
    can reach any other op in the group. Return True if it can.
    """
    assert (
        grouped_op in group
    ), f'grouped_op {grouped_op._attrs["name"]} is not from the group'
    for op in group:
        if op is grouped_op:
            continue
        if transform_utils.is_ancestor(op, grouped_op):
            return True
    return False


def _group_split_outputs_together(
    sorted_graph: List[Tensor], sorted_ops: List[Operator], op_type: str
) -> List[List[Operator]]:
    """As long as alignment allows, we group all output gemm ops from split op
    together to eliminate the cost of split. Here we don't exclude large gemms
    because the copy of split is worse than group gemm overhead.
    """
    groups = []
    if op_type not in _group_gemm_op_mapping:
        return groups
    for op in sorted_ops:
        if op._attrs["op"] != "split":
            continue
        if not _can_fuse_split_op(op):
            continue

        gemm_group = []
        for output in op._attrs["outputs"]:
            dst_ops = list(output.dst_ops())
            if len(dst_ops) == 0:
                break
            gemm_op = dst_ops[0]
            if gemm_op._attrs["op"] != op_type:
                break
            if not _filter_gemm_op(gemm_op):
                break
            if not gemm_group:
                gemm_group.append(gemm_op)
            else:
                if _gemm_op_check(gemm_group[-1], gemm_op):
                    gemm_group.append(gemm_op)
                else:
                    break
        if len(gemm_group) == len(op._attrs["outputs"]) and all(
            not _has_cycle(grouped_op, gemm_group) for grouped_op in gemm_group
        ):
            _fuse_gemm_ops(gemm_group, sorted_graph)
            groups.append(gemm_group)
    return groups


def _dfs(
    tensor: Tensor, op_type: str, visited: OrderedDict[Tensor, Set[Operator]]
) -> Set[Operator]:
    """Dfs pass to traverse the graph and collects descendant ops with type == op_type
    for every tensor, which is saved in `visited`.
    """
    if tensor in visited:
        return visited[tensor]

    # establish topological order in visited
    visited[tensor] = set()

    descendants = set()
    for op in tensor.dst_ops():
        outputs = op._attrs["outputs"]
        for output in outputs:
            descendants.update(_dfs(output, op_type, visited))

    # visited[tensor] should only contain descendants, not self
    visited[tensor].update(descendants)

    src_ops = list(tensor.src_ops())
    assert (
        len(src_ops) <= 1
    ), f"A tensor can't have more than 1 src_op in this stage, len(src_ops): {len(src_ops)}"

    src_op = src_ops[0] if len(src_ops) == 1 else None
    if src_op and src_op._attrs["op"] == op_type:
        descendants.add(src_op)
    return descendants


def _filter_by_op_type(
    visited: OrderedDict[Tensor, Set[Operator]], op_type: str
) -> OrderedDict[Operator, Set[Operator]]:
    """Go through `visited` and ony save the entries with parent op == op_type in
       the final dependency graph

    Args:
        visited (OrderedDict[Tensor, Set[Operator]]): {tensor: descendants of tensor} pairs
            for all tensors in the graph
        op_type (str): The op type to be grouped

    Returns:
        OrderedDict[Operator, Set[Operator]]: The final dependency graph
    """
    final = collections.OrderedDict()
    for parent, descendants in visited.items():
        if parent.src_ops():
            src_op = list(parent.src_ops())[0]
            parent_op_type = src_op._attrs["op"]
            if parent_op_type == op_type:
                final[src_op] = descendants
    return final


def _get_dependency_graph(
    sorted_graph: List[Tensor], op_type: str
) -> OrderedDict[Operator, Set[Operator]]:
    """Get dependency graph: `G: {op: all the descendants of op with op_type}`.
       G only contains ops with type == op_type.
       The dependency doesn't necessarily need to be topologically sorted.
       It's helpful for debugging though. So let's keep it ordered.

    Args:
        sorted_graph (List[Tensor]): Topologically sorted graph
        op_type (str): The op type to be grouped

    Returns:
        OrderedDict[Operator, Set[Operator]]: The dependency graph.

    Algorithm:
    1) Do dfs to traverse the graph and collects descendants ops with type == op_type
       for every tensor, which is saved in `visited`.
    2) Go through `visited` and ony save the entries with parent op == op_type in
       the final dependency graph
    """
    visited = collections.OrderedDict()
    for tensor in sorted_graph:
        _dfs(tensor, op_type, visited)

    filtered = _filter_by_op_type(visited, op_type)

    return filtered


def _get_sorted_candidate_ops(
    sorted_ops: List[Operator], op_type: str, f_filter: Callable
) -> OrderedDict[Tensor, bool]:
    """Get all the candidate ops, `grouped: {op: flag}`, for group fusion. The flag
       denotes whether this op is grouped or not and is initialized to False. We need to
       filter out ops that are not eligible such as gemm ops with large m/n/k or odd
       alignment, or layernorm ops without gamma/beta etc.

    Args:
        sorted_ops (List[Operator]): Sorted ops from the graph
        op_type (str): The op type to be grouped

    Returns:
        OrderedDict[Tensor, bool]: All the candidate ops for group fusion
    """
    op_set = collections.OrderedDict()
    for op in sorted_ops:
        if op._attrs["op"] == op_type and f_filter(op):
            op_set[op] = False

    return op_set


# 39 comes from the kernel requirement, for > 39 groups, we need to copy
# the arguments to gpu memory with sync memcpy, which is bad for perf
_MAX_LAYERNORM_GROUP = 39


# TODO: remove after switching to async copy for group layernorm args
def _break_layernorm_groups(group: List[Operator]) -> List[List[Operator]]:
    if len(group) <= _MAX_LAYERNORM_GROUP:
        return group
    group.sort(key=lambda x: _get_layernorm_flattened_normalized_shape(x), reverse=True)
    groups = []
    num_groups = (len(group) + _MAX_LAYERNORM_GROUP - 1) // _MAX_LAYERNORM_GROUP

    for i in range(num_groups):
        begin = i * _MAX_LAYERNORM_GROUP
        end = min((i + 1) * _MAX_LAYERNORM_GROUP, len(group))
        groups.append(group[begin:end])
    return groups


def _group_ops_by_type(
    sorted_graph: List[Tensor], op_type: str, workdir: str = None
) -> bool:
    """Find and fuse all groups of ops that can be fused together.
    Each group is replaced with 1 group op.

    Args:
        sorted_graph (List[Tensor]): Topologically sorted input graph
        op_type (str): The type of op to be grouped

    Returns:
        True if we fused any group.

    The algorithm can be described as:
    0) Let groups = []
    1) Do dfs and get the dependency graph, `G: {op: all the descendants of op with op_type}`.
       G only contains ops with type == op_type. G does not need to be in topological order.
    2) Get all the candidate ops, `grouped: {op: flag}`, for group fusion. The flag
       denotes whether this op is grouped or not and is initialized to False. We need to
       filter out ops that are not eligible such as gemm ops with large m/n/k or odd
       alignment, or layernorm ops without gamma/beta etc. Later, we set all grouped ops
       in grouped to True. Ops that can't be grouped with any other ops are also set to True.
       **Ops in `grouped` must be in topological order.**
    3) For group gemm, we group all gemms following the same split op together so the
       split op can be eliminated. Due to the high cost of split, we don't apply the same
       m/n/k filter to these gemm ops. These ops are removed from `grouped`.
    4) Let op_set = set(grouped.keys()), all the ops available for grouping.
    5) For every op in grouped:
            If grouped[candidate] is True, continue to next op.
            Remove op from op_set. Because ops in grouped are topologically sorted, this
                guarantees that op_set won't contain any ancestors of op.
            Get op candidates that can be potentially grouped with op.
            candidates = op_set - {G[op]} where G[op] is descendant of op.
            Sort candidates by name (same as topological order).
            for every candidate in candidates:
                Check if op and candidate op can be grouped together.
                    - Check for dependency
                    - Check for op compatibility
                If yes:
                    Group them together and remove candidate from op_set.
                    Merge descendants of candidate op to descendants of op.
                    Set grouped[candidate] = True
            If the final group is >= 2, add them to `groups`
        Set grouped[op] = True
    """

    # TODO: as an optimization, we may keep using the same dependency_graph
    # through all the group passes and keep updating it
    dependency_graph = _get_dependency_graph(sorted_graph, op_type)

    # There is no op with op_type in the graph
    if len(dependency_graph) == 0:
        return False

    if workdir:
        _dump_dependency_graph(dependency_graph, op_type, "filtered", workdir)

    f_filter_op = _get_op_filter(op_type)
    f_check_ops_are_compatible = _get_op_checker(op_type)
    is_layernorm = op_type.startswith("layernorm")
    f_fuse_ops = _fuse_layernorm_ops if is_layernorm else _fuse_gemm_ops

    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)

    # grouped: {key: op, value: whether this op has been grouped or not}
    # ops in grouped must be in topological order
    grouped = _get_sorted_candidate_ops(sorted_ops, op_type, f_filter_op)
    assert len(grouped) <= len(dependency_graph)

    groups = []

    # applies to group gemms only
    split_groups = _group_split_outputs_together(sorted_graph, sorted_ops, op_type)
    for group in split_groups:
        groups.append(group)
        for op in group:
            if op in grouped:
                del grouped[op]

    # the set of ops available for group fusion
    op_set = set(grouped.keys())

    for op, visited in grouped.items():
        if visited:
            op_set.discard(op)
            continue
        descendants = dependency_graph[op]
        op_set.discard(op)

        candidates = op_set - descendants

        def get_op_number(op: Operator) -> int:
            op_name = op._attrs["name"]
            last_idx = op_name.rfind("_")
            return int(op_name[last_idx + 1 :])

        # Sort by topological order. Sorting by names guarantees topological order
        # because of how the name_graph pass works
        group_candidates = sorted(candidates, key=lambda x: get_op_number(x))

        group = [op]
        for candidate in group_candidates:
            if candidate in descendants:
                continue

            if (
                _ops_have_same_num_inputs(op, candidate)
                and _check_op_num_outputs(op, 1)
                and f_check_ops_are_compatible(op, candidate)
            ):
                group.append(candidate)
                grouped[candidate] = True

                op_set.discard(candidate)

                # must merge descendants together
                descendants.update(dependency_graph[candidate])
        # remove any op that may introduce a cycle because of grouping ops
        group_op_idx = 0
        while group_op_idx < len(group):
            grouped_op = group[group_op_idx]
            if _has_cycle(grouped_op, group):
                del group[group_op_idx]
            else:
                group_op_idx += 1

        # We fuse each group right after we form it. Otherwise, _has_cycle may
        # miss cycles within groups. For example, see the graph below:
        #
        #        A --> C ---
        #                  |
        #    --> B --> D   |
        #    |             |
        #    --- X --> M   |
        #                  |
        #        Y --> N <--
        #
        # If we fuse (A, B) and (X, Y) at the same time, we would end up with a
        # cycle between the fused op (A, B) and (X, Y). On the other hand, if we
        # fuse (A, B) first, and then check _has_cycle before fusing (X, Y), we
        # will be able to detect the cycle.
        if len(group) > _MAX_LAYERNORM_GROUP and op_type.startswith("layernorm"):
            new_groups = _break_layernorm_groups(group)
            for new_group in new_groups:
                f_fuse_ops(new_group, sorted_graph)
            groups.extend(new_groups)
        elif len(group) >= 2:
            f_fuse_ops(group, sorted_graph)
            groups.append(group)

        grouped[op] = True

    if workdir:
        _dump_groups(groups, op_type, workdir)

    return len(groups) > 0


def _fuse_layernorm_ops(
    op_group: List[Operator], sorted_graph: List[Tensor]
) -> List[Tensor]:
    """
    Replace a group of ops with a single group op
    """
    # Make the order deterministic
    # Sort by gamma name
    op_group.sort(key=lambda x: x._attrs["inputs"][1]._attrs["name"])

    # gather inputs
    num_inputs = len(op_group[0]._attrs["inputs"])
    group_inputs = [[] for _ in range(num_inputs)]
    normalized_shapes = []
    for op in op_group:
        normalized_shapes.append(op._attrs["normalized_shape"])
        for i, input in enumerate(op._attrs["inputs"]):
            group_inputs[i].append(input)

    # remove dst_ops from inputs
    for op in op_group:
        for input in op._attrs["inputs"]:
            transform_utils.remove_dst_op_from_tensor(input, op)

    # create group op
    op_type = op_group[0]._attrs["op"]
    group_op = (
        ops.group_layernorm
        if op_type == "layernorm"
        else ops.group_layernorm_sigmoid_mul
    )
    eps = op_group[0]._attrs["eps"]
    group_outputs = group_op()(
        group_inputs[0], group_inputs[1], group_inputs[2], normalized_shapes, eps
    )

    for i, op in enumerate(op_group):
        new_output = group_outputs[i]
        op_output = op._attrs["outputs"][0]
        transform_utils.replace_tensor(op_output, new_output)

    # sorted_graph is no longer sorted here
    sorted_graph.extend(group_outputs)
    return sorted_graph


def _fuse_gemm_ops(
    op_group: List[Operator], sorted_graph: List[Tensor]
) -> List[Tensor]:
    """
    Replace a group of ops with a single group op
    """

    assert op_group[0]._attrs["op"].startswith("gemm_rcr"), (
        f"_fuse_gemm_ops only supports gemm_rcr family ops. "
        f"{op_group[0]._attrs['op']} is not supported"
    )

    # Sort by weight name
    op_group.sort(key=lambda x: x._attrs["inputs"][1]._attrs["name"])

    # Make the order deterministic, important for cache hit.
    # sort ops by decreasing K, N
    op_group.sort(
        key=lambda x: (
            x._attrs["inputs"][1]._size(1).value(),
            x._attrs["inputs"][1]._size(0).value(),
        ),
        reverse=True,
    )

    # gather inputs
    group_inputs = [op._attrs["inputs"] for op in op_group]

    # remove dst_ops from inputs
    for op in op_group:
        for input in op._attrs["inputs"]:
            transform_utils.remove_dst_op_from_tensor(input, op)

    # create group op
    op_type = op_group[0]._attrs["op"]
    assert op_type in _group_gemm_op_mapping, f"{op_type} not in _group_gemm_op_mapping"
    group_op = _group_gemm_op_mapping[op_type]

    group_outputs = group_op()(group_inputs)

    for i, op in enumerate(op_group):
        new_output = group_outputs[i]
        op_output = op._attrs["outputs"][0]
        transform_utils.replace_tensor(op_output, new_output)

    # sorted_graph is no longer sorted here
    sorted_graph.extend(group_outputs)
    return sorted_graph


# TODO: add slice + group_gemm fusion
def _fuse_group_ops_by_type(
    sorted_graph: List[Tensor], op_type: str, workdir: str = None
) -> List[Tensor]:
    """
    This pass groups gemm ops or layernorm ops together.

    For gemm ops, the supported op types are:
    - gemm_rcr
    - gemm_rcr_bias
    - gemm_rcr_bias_relu
    - gemm_rcr_bias_sigmoid.

    There are several conditions that the ops must meet to be grouped together:
    1) alignment requirement: all ops in a group must have the same ab_alignment.
       Minimal ab_alignment required is 2.
       TODO: experiment with grouping ops with different alignment together and use
       minimal alignment. Need to benchmark perf.
    2) ops must have the same type and alpha = 1.0. Only 2D gemms are supported.
    3) all gemm ops after a single split op will be grouped together into one group
       regardless of condition 4
    4) TODO: exclude gemm ops with large m/n/k with heuristics
    6) TODO: support dynamic batch dim. All dims must be static currently.

    For layernorm ops, the supported op types are:
    - layernorm
    - layernorm_sigmoid_mul

    The group conditions are
    1) all ops must have the same eps
    2) neither gamma or beta can be None
    3) the normalized_shape must have same rank
    4) all inputs must have the same rank and batch dimensions
    5) all inputs must have the same alignment according to _get_layernorm_alignment

    The overall algorithm is pretty simple. It takes two steps:
    1) find all groups of op to fuse together
    2) fuse them together
    Details of step 1 can be found in _group_ops_by_type
    """
    # if we didn't fuse any grouped ops, we simply return original sorted_graph
    if not _group_ops_by_type(sorted_graph, op_type, workdir):
        return sorted_graph

    sorted_graph = toposort(sorted_graph)
    sorted_graph = transform_utils.sanitize_sorted_graph(sorted_graph)
    return sorted_graph


def fuse_group_gemm_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    # gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_relu, gemm_rcr_bias_sigmoid
    for op_type in _group_gemm_op_mapping.keys():
        sorted_graph = _fuse_group_ops_by_type(sorted_graph, op_type, workdir)
    return sorted_graph


def fuse_group_layernorm_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    for op_type in ["layernorm_sigmoid_mul", "layernorm"]:
        sorted_graph = _fuse_group_ops_by_type(sorted_graph, op_type, workdir)
    return sorted_graph


# The right order for graph passes is:
# fuse_mm_elementwise, to prefer elementwise epilogue fusions (better overall perf)
# fuse_group_ops,
# fuse_strided_ops, (need to add more group gemm fusion passes)
def fuse_group_ops(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """Horizontal fusion of grouped gemm and layernorm ops

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str, optional
        working dir, by default None

    Returns
    -------
    List[Tensor]
        New graph after fusion
    """
    # gemms need to be fused first
    # TODO: enable after adding heuristics and fixing dynamic shapes
    from aitemplate.backend.target import Target

    if Target.current().name() == "cuda":
        if "fuse_group_gemm" in Target.current()._kwargs:
            sorted_graph = fuse_group_gemm_ops(sorted_graph)
        sorted_graph = fuse_group_layernorm_ops(sorted_graph)

    return sorted_graph
