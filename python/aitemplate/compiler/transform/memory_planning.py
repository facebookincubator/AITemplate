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
Graph pass for memory planning.
"""
import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils.environ import multistream_max_mem_parallel_ops, multistream_mode
from aitemplate.utils.graph_utils import split_simple_multistream_parallel_ops

# pylint: disable=C0103

_LOGGER = logging.getLogger(__name__)


@dataclass
class TensorUsageRecord:
    """
    A named tuple to keep a tensor usage record, where

    tensor: this tensor

    first_op_idx: the index of the first op that uses this tensor as its input
                  or output

    last_op_idx: the index of the last op that uses this tensor as its input or
                 output

    size: the size of this tensor
    """

    tensor: Tensor
    first_op_idx: int
    last_op_idx: int
    size: int

    def __iter__(self):
        return iter([self.tensor, self.first_op_idx, self.last_op_idx, self.size])


def _find_original_tensor(tensor: Tensor):
    """Find the original tensor of a tensor view recursively."""
    view = tensor._attrs["is_view_of"]
    if not view:
        return tensor
    return _find_original_tensor(view)


def _make_tensor_usage_records(sorted_ops: List[Operator]) -> List[TensorUsageRecord]:
    num_of_ops = len(sorted_ops)
    tensor_records = defaultdict(
        lambda: TensorUsageRecord(
            tensor=None, first_op_idx=num_of_ops, last_op_idx=-1, size=None
        )
    )
    for op_idx, op in enumerate(sorted_ops):
        for tensor in op._attrs["inputs"] + op._attrs["outputs"]:
            # Skip weights and inputs since we don't overwrite them.
            # Note that it might be OK to overwrite inputs, but let's be
            # consertative for now and not surprise users. We could always
            # make a flag to do that later if it's needed.
            if tensor._attrs["is_param"]:
                continue
            name = tensor._attrs["name"]
            this_tensor = tensor_records[name].tensor
            if this_tensor is None:
                tensor_records[name].tensor = tensor
            else:
                # make sure we didn't screw up anything
                assert (
                    tensor == this_tensor
                ), f"existing tensor: {this_tensor}, new tensor: {tensor}, op: {op}"

            first_op_idx = tensor_records[name].first_op_idx
            last_op_idx = tensor_records[name].last_op_idx
            tensor_records[name].first_op_idx = min(first_op_idx, op_idx)
            tensor_records[name].last_op_idx = max(last_op_idx, op_idx)
            # An output tensor's lifetime extends to the last op.
            if tensor._attrs["is_output"]:
                tensor_records[name].last_op_idx = num_of_ops - 1

            size = tensor_records[name].size
            tensor_size = tensor.size_bytes(alignment=64)
            if size is None:
                tensor_records[name].size = tensor_size
            else:
                # make sure we didn't screw up anything
                assert size == tensor_size

    # tensor views extend the lifetime of the original tensors
    tensor_views = []
    for name, tensor_record in tensor_records.items():
        this_tensor = tensor_record.tensor
        if this_tensor._attrs["is_view_of"]:
            orig_tensor = _find_original_tensor(this_tensor)
            # view of input
            if orig_tensor._attrs["is_param"]:
                continue
            orig_tensor_name = orig_tensor._attrs["name"]
            assert orig_tensor_name in tensor_records
            tensor_records[orig_tensor_name].last_op_idx = max(
                tensor_records[orig_tensor_name].last_op_idx, tensor_record.last_op_idx
            )
            tensor_views.append(name)

    # remove tensor views from tensor_records
    for name in tensor_views:
        del tensor_records[name]

    # sanity checks
    # make sure we have valid indices and sizes
    records = tensor_records.values()
    for tensor, first_op_idx, last_op_idx, size in records:
        assert tensor is not None
        assert 0 <= first_op_idx < num_of_ops
        assert 0 <= last_op_idx < num_of_ops
        assert first_op_idx <= last_op_idx
        assert size is not None

    return list(records)


def assign_offsets_to_views_and_outputs(sorted_graph: List[Tensor]) -> None:
    """Propagate offsets determined by the memory planning algorithm to views.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        The graph, modified in-place
    """
    for node in sorted_graph:
        if node._attrs["is_view_of"]:
            node._attrs["offset"] = node._attrs["is_view_of"]._attrs["offset"]


@dataclass
class Workspace:
    shared_size: int
    unique_size: int

    def total_size(self) -> int:
        return self.shared_size + self.unique_size


def _compute_workspace(sorted_graph: List[Tensor]) -> Workspace:
    """
    Compute the workspace for the model, which can be used as scratch memory by ops.
    This pass examines two attributes on every function in the graph:
    - workspace: The amount of memory in bytes to be used as shared scratch memory.
      Here, "shared" means that other ops are allowed to write to this memory.
    - unique_workspace: The amount of memory in bytes to be used as exclusive scratch memory.
      If set, this pass will assign the op a "unique_workspace_offset". This can be used at
      codegen time to set a pointer to the region of exclusive shared memory.

    The returned Workspace has two attributes:
    - shared_size: The total memory needed for all op's shared scratch memory (i.e. the maximum
    of all workspace attributes)
    - unique_size: The total memory needed for all unique scratch memory (i.e. the sum of
    all unique_workspace attributes)

    During codegen, the workspace gets set up like this:
    [--unique 1--][--unique 2--]...[--unique N--][--shared--]
    """
    unique_workspace_size = 0
    max_workspace = 0
    for node in sorted_graph:
        for func in node._attrs["src_ops"]:
            if "workspace" in func._attrs:
                max_workspace = max(max_workspace, func._attrs["workspace"])
            if (
                "unique_workspace" in func._attrs
                and "unique_workspace_offset" not in func._attrs
            ):
                func._attrs["unique_workspace_offset"] = unique_workspace_size
                unique_workspace_size += func._attrs["unique_workspace"]
    return Workspace(max_workspace, unique_workspace_size)


def _greedy_by_size_memory_planning(
    sorted_graph: List[Tensor], tensor_usage_records: List[TensorUsageRecord]
):
    """
    based on the greedy-by-size algorithm for offset calculation described in
    the following paper:
        Yury Pisarchyk, Juhyun Lee,
        Efficient Memory Management for Deep Neural Net Inference,
        https://arxiv.org/abs/2001.03288
    """
    # sort tensor usage records in non-increasing order by their sizes
    sorted_tensor_usage_records = sorted(
        tensor_usage_records, key=lambda r: r.size, reverse=True
    )

    max_blob = 0
    # For tensors that have been assigned, we keep their tensor usage records
    # in increasing order by memory offsets
    sorted_assigned_records = []
    for tensor_record in sorted_tensor_usage_records:
        tensor, first_op_idx, last_op_idx, size = tensor_record
        prev_offset = 0
        best_offset = None
        smallest_gap = pow(2, 63) - 1
        # Iterate through tensors that have been allocated.
        # For those whose usage intervals intersect with that of current
        # tensor, we try to find the smallest valid memory gap between such two
        # allocated tensors, which is big enough to hold current tensor.
        # If such a gap is found, we will place current tensor in the gap.
        for a_record in sorted_assigned_records:
            a_tensor, a_first_op_idx, a_last_op_idx, a_size = a_record
            max_first_op_idx = max(first_op_idx, a_first_op_idx)
            min_last_op_idx = min(last_op_idx, a_last_op_idx)
            # current tensor overlaps with this assigned tensor
            if max_first_op_idx <= min_last_op_idx:
                a_offset = a_tensor._attrs["offset"]
                gap = a_offset - prev_offset
                if size <= gap < smallest_gap:
                    smallest_gap = gap
                    best_offset = prev_offset
                prev_offset = max(prev_offset, a_offset + a_size)
        # If we can't find a valid memory gap between two allocated tensors,
        # we put current tensor to the rightmost tensor whose usage interval
        # intersects with that of the current tensor.
        if best_offset is None:
            best_offset = prev_offset
        tensor._attrs["offset"] = best_offset
        max_blob = max(max_blob, best_offset + size)

        # bisect from Python <=3.9 doesn't have the key parameter
        sorted_offsets = [r.tensor._attrs["offset"] for r in sorted_assigned_records]
        in_pos = bisect.bisect_right(
            sorted_offsets, tensor_record.tensor._attrs["offset"]
        )
        sorted_assigned_records.insert(in_pos, tensor_record)

    # now we assign blobs for weights and inputs
    constant_offset = 0
    for node in sorted_graph:
        if (
            node._attrs["data"] is not None
            or node._attrs["constant_folding_output_idx"] is not None
        ):
            node._attrs["offset"] = constant_offset
            constant_offset += node.size_bytes(alignment=64)

    # assign offsets to tensor views
    # this step must happen after weights and inputs are assigned so that views
    # of inputs are properly handled
    assign_offsets_to_views_and_outputs(sorted_graph)

    workspace = _compute_workspace(sorted_graph)

    # make sure we've covered the entire graph
    return (max_blob, constant_offset, workspace)


def greedy_by_size_memory_planning(sorted_graph: List[Tensor]):  # noqa: C901
    """
    based on the greedy-by-size algorithm for offset calculation described in
    the following paper:
        Yury Pisarchyk, Juhyun Lee,
        Efficient Memory Management for Deep Neural Net Inference,
        https://arxiv.org/abs/2001.03288
    """
    sorted_ops = []
    for node in sorted_graph:
        sorted_ops.extend(node.src_ops())
    tensor_usage_records = _make_tensor_usage_records(sorted_ops)

    return _greedy_by_size_memory_planning(sorted_graph, tensor_usage_records)


def naive_memory_planning(sorted_graph: List[Tensor]):
    max_blob = 0
    offset = 0
    constant_offset = 0
    for node in sorted_graph:
        if (
            node._attrs["data"] is not None
            or node._attrs["constant_folding_output_idx"] is not None
        ):
            node._attrs["offset"] = constant_offset
            constant_offset += node.size_bytes(alignment=64)
        elif not node._attrs["is_view_of"]:
            node._attrs["offset"] = offset
            tensor_size = node.size_bytes(alignment=64)
            offset += tensor_size
            max_blob += tensor_size

    # workspace
    workspace = _compute_workspace(sorted_graph)
    assign_offsets_to_views_and_outputs(sorted_graph)

    return (max_blob, constant_offset, workspace)


def _make_tensor_usage_records_simple_multistream(
    par_ops_seq: List[List[Operator]],
) -> List[TensorUsageRecord]:
    """
    Generalized version of _make_tensor_usage_records() which
    assumes that several ops may be executed on every step.

    Simple multistream algo iteratively tracks sets of operators
    that can be run in parallel independently on each iteration.

    par_ops_seq contains lists of operators that can be run
    in parallel on every algorithm iteration.

    Technically, the regular _make_tensor_usage_records() version
    is similar to the following one:

       def _make_tensor_usage_records(sorted_ops):
         par_ops_seq = [sorted_ops]
         return _make_tensor_usage_records_simple_multistream(par_ops_seq)

    This version is kept as a separate one, because multistreaming
    feature is still somewhat experimental.
    """

    num_of_ops = len(par_ops_seq)
    tensor_records = defaultdict(
        lambda: TensorUsageRecord(
            tensor=None, first_op_idx=num_of_ops, last_op_idx=-1, size=None
        )
    )

    for op_idx, par_ops in enumerate(par_ops_seq):
        for op in par_ops:
            for tensor in op._attrs["inputs"] + op._attrs["outputs"]:
                # Skip weights and inputs since we don't overwrite them.
                # Note that it might be OK to overwrite inputs, but let's be
                # consertative for now and not surprise users. We could always
                # make a flag to do that later if it's needed.
                if tensor._attrs["is_param"]:
                    continue
                name = tensor._attrs["name"]
                this_tensor = tensor_records[name].tensor
                if this_tensor is None:
                    tensor_records[name].tensor = tensor
                else:
                    # make sure we didn't screw up anything
                    assert (
                        tensor == this_tensor
                    ), f"existing tensor: {this_tensor}, new tensor: {tensor}, op: {op}"

                first_op_idx = tensor_records[name].first_op_idx
                last_op_idx = tensor_records[name].last_op_idx
                tensor_records[name].first_op_idx = min(first_op_idx, op_idx)
                tensor_records[name].last_op_idx = max(last_op_idx, op_idx)
                # An output tensor's lifetime extends to the last op.
                if tensor._attrs["is_output"]:
                    tensor_records[name].last_op_idx = num_of_ops - 1

                size = tensor_records[name].size
                tensor_size = tensor.size_bytes(alignment=64)
                if size is None:
                    tensor_records[name].size = tensor_size
                else:
                    # make sure we didn't screw up anything
                    assert size == tensor_size

    # tensor views extend the lifetime of the original tensors
    tensor_views = []
    for name, tensor_record in tensor_records.items():
        this_tensor = tensor_record.tensor
        if this_tensor._attrs["is_view_of"]:
            orig_tensor = _find_original_tensor(this_tensor)
            # view of input
            if orig_tensor._attrs["is_param"]:
                continue
            orig_tensor_name = orig_tensor._attrs["name"]
            assert orig_tensor_name in tensor_records
            tensor_records[orig_tensor_name].last_op_idx = max(
                tensor_records[orig_tensor_name].last_op_idx, tensor_record.last_op_idx
            )
            tensor_views.append(name)

    # remove tensor views from tensor_records
    for name in tensor_views:
        del tensor_records[name]

    # sanity checks
    # make sure we have valid indices and sizes
    records = tensor_records.values()
    for tensor, first_op_idx, last_op_idx, size in records:
        assert tensor is not None
        assert 0 <= first_op_idx < num_of_ops
        assert 0 <= last_op_idx < num_of_ops
        assert first_op_idx <= last_op_idx
        assert size is not None

    return list(records)


def simple_multistream_memory_planning(sorted_graph: List[Tensor]):
    """
    A specialized case for simple multi-stream execution.
    It uses more or slightly more GPU memory than greedy_by_size_memory_planner,
    depending on the input graph, but still significantly less
    than naive_memory_planning.
    """
    from aitemplate.utils.graph_utils import track_graph_timings

    # track the sequence
    time_stats = track_graph_timings(sorted_graph, {})

    # sort all operators by parallel execution order
    ops_by_order = defaultdict(list)
    for (op, tracking) in time_stats.op_parallel_trackers.items():
        ops_by_order[tracking.execution_order].append(op)

    # convert Dict[int, List[Operator]] into List[List[Operator]]
    max_parallel_ops = multistream_max_mem_parallel_ops()
    par_ops_seq = split_simple_multistream_parallel_ops(ops_by_order, max_parallel_ops)

    tensor_usage_records = _make_tensor_usage_records_simple_multistream(par_ops_seq)

    return _greedy_by_size_memory_planning(sorted_graph, tensor_usage_records)


def proxy_memory_planning(sorted_graph: List[Tensor]):
    run_mode = multistream_mode()
    if run_mode == 0:
        # no multistream
        max_blob, constant_offset, workspace = greedy_by_size_memory_planning(
            sorted_graph
        )
    elif run_mode == 1:
        # simple multistream
        max_blob, constant_offset, workspace = simple_multistream_memory_planning(
            sorted_graph
        )
    else:
        # unsupported
        raise Exception(f"Unsupported multistream mode ({run_mode})")

    # print some statistics
    _LOGGER.info(
        f"Workspace shared_size={workspace.shared_size} unique_size={workspace.unique_size}"
    )
    _LOGGER.info(f"max_blob={max_blob} constant_offset={constant_offset}")

    # done
    return (max_blob, constant_offset, workspace)


# memory_planning = greedy_by_size_memory_planning
# memory_planning = naive_memory_planning
memory_planning = proxy_memory_planning
