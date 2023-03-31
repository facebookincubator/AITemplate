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
import json
import logging
import os
from collections import deque
from typing import Any, Dict, List, Set

from aitemplate.utils.misc import is_debug
from aitemplate.utils.visualization import plot_graph


_LOGGER = logging.getLogger(__name__)


def get_sorted_ops(tensors) -> List[Any]:
    """
    Produces the exact execution sequence of operators.
    This matches backend/codegen.py, ModelContainerGenerator.append_all_tensors()
    """

    from aitemplate.compiler.base import Tensor

    visited = set()
    sorted_ops = []
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for tensor in tensors:
        for src_op in tensor.src_ops():
            if src_op in visited:
                continue
            visited.add(src_op)
            sorted_ops.append(src_op)
    return sorted_ops


def sorted_graph_debug_str(tensors) -> str:
    from aitemplate.compiler.base import Tensor

    if isinstance(tensors, Tensor):
        tensors = [tensors]
    tensor_str = "\n\n".join([str(tensor) for tensor in tensors])
    op_str = "\n\n".join([str(op) for op in get_sorted_ops(tensors)])
    return "Tensors: {}\n\nOperators: {}\n\n".format(tensor_str, op_str)


def sorted_graph_debug_json(tensors) -> str:
    from aitemplate.compiler.base import Tensor
    from aitemplate.utils.json_utils import gen_unique_op_names, GraphJsonEncoder

    if isinstance(tensors, Tensor):
        tensors = [tensors]

    json_dict = {}
    json_dict["Tensors"] = tensors
    json_dict["Operators"] = get_sorted_ops(tensors)

    op_names = gen_unique_op_names(tensors)
    encoder = GraphJsonEncoder(op_names, indent=2)

    return encoder.encode(json_dict)


def sorted_graph_pseudo_code(tensors, with_shape=True) -> str:
    from aitemplate.compiler.base import Tensor

    if isinstance(tensors, Tensor):
        tensors = [tensors]
    op_str = "\n".join([op.pseudo_code(with_shape) for op in get_sorted_ops(tensors)])
    return op_str


def sorted_op_pseudo_code(ops, with_shape=True) -> str:
    from aitemplate.compiler.base import Operator

    if isinstance(ops, Operator):
        ops = [ops]
    op_str = "\n".join([op.pseudo_code(with_shape) for op in ops])
    return op_str


def dump_graph_debug_str_to_file(tensors, workdir, name, file_with_time_profiles=None):
    if is_debug():
        # Dump graph and pseudo code for debug only
        debug_path = workdir + "/debug"
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        prefix = os.path.join(debug_path, name)
        graph_path = prefix + "_graph.txt"
        graph_json_path = prefix + "_graph.json"
        pseudo_code_path = prefix + "_pseudo_code.txt"
        graph_visual_path = prefix + "_graph_vis.html"
        with open(graph_path, "w") as f:
            f.write(sorted_graph_debug_str(tensors))
            _LOGGER.debug(f"Dumped {name} graph to {graph_path}")
        with open(graph_json_path, "w") as f:
            f.write(sorted_graph_debug_json(tensors))
            _LOGGER.debug(f"Dumped {name} graph to {graph_json_path}")
        with open(pseudo_code_path, "w") as f:
            f.write(sorted_graph_pseudo_code(tensors))
            _LOGGER.debug(f"Dumped {name} pseudo code to {pseudo_code_path}")
        plot_graph(tensors, graph_visual_path, file_with_time_profiles)
        _LOGGER.debug(f"Dumped {name} visualization to {graph_visual_path}")


class TimestampTracking:
    def __init__(
        self, execution_start: float = 0, duration: float = 0, execution_order: int = 0
    ):
        self.execution_order = execution_order
        self.execution_start = execution_start
        self.duration = duration

    @property
    def execution_end(self):
        return self.execution_start + self.duration


class ProfiledTimeStatistics:
    def __init__(self):
        # Dict[Operator, float]
        self.op_durations = {}

        # Dict[Operator, TimestampTracking]
        self.op_parallel_trackers = {}
        # Dict[Operator, TimestampTracking]
        self.op_sequential_trackers = {}

        # Dict[Trnsor, TimestampTracking]
        self.tensor_parallel_trackers = {}
        # Dict[Trnsor, TimestampTracking]
        self.tensor_sequential_trackers = {}

        # 0.7 percentile of op times
        self.duration_p70 = 0.0
        # 0.9 percentile of op times
        self.duration_p90 = 0.0
        # 0.95 percentile of op times
        self.duration_p95 = 0.0
        # max time spent among operators
        self.duration_max = 0.0
        # total time spent by operators
        self.total_duration = 0.0


def track_graph_timings(
    tensors, file_path_profiler_output: str
) -> ProfiledTimeStatistics:
    """
    Traverses the graph of tensors and uses the statistics from the profiler
    to evaluate execution times in case of sequential execution (1 stream)
    and parallel execution (unlimited number of streams).
    """

    from aitemplate.compiler.base import Operator, Tensor

    output = ProfiledTimeStatistics()

    # the exact sequence of non-constant tensors that need to be evaluated
    #   within a single execution stream.
    unprocessed_tensors: List[Tensor] = []

    # Sequence_of_ops contains an exact execution sequence of ops
    #   within a single execution stream.
    # Similar to graph_utils.py, get_sorted_ops() call.
    sequence_of_ops: List[Operator] = []
    visited_ops: Set[Operator] = set()

    for tensor in tensors:
        src_ops = tensor.src_ops()

        if len(src_ops) == 0:
            # This tensor depends on no operator.
            # So, add the final statistics for it.
            output.tensor_parallel_trackers[tensor] = TimestampTracking()
            output.tensor_sequential_trackers[tensor] = TimestampTracking()
        else:
            for op in src_ops:
                if op not in visited_ops:
                    visited_ops.add(op)
                    sequence_of_ops.append(op)

            # this tensor needs to be evaluated
            unprocessed_tensors.append(tensor)

    # ok, we've got ops. Load the file with the profile.
    with open(file_path_profiler_output, "r") as f:
        perf_per_op_str = f.read()

    # parse file
    perf_per_op_str_dict = json.loads(perf_per_op_str)

    op_durations: Dict[str, float] = {}
    for op_name, op_data in perf_per_op_str_dict.items():
        op_durations[op_name] = op_data["ms_per_iter"]

    # map timings to ops
    for op in visited_ops:
        # profiler records the results under the original_name
        op_name = op._attrs["original_name"]

        # replace op_name with a unique name, if provided
        if op_name is not None:
            if op_name not in op_durations:
                # op_name was not found in the profiler report
                output.op_durations[op] = 0
            else:
                time_cost = op_durations[op_name]
                output.op_durations[op] = time_cost
        else:
            # op_name is None, idk what to do
            output.op_durations[op] = 0

    # compute statistics
    sorted_op_durations = sorted(op_durations.values())
    if len(sorted_op_durations) > 0:
        output.duration_p70 = sorted_op_durations[int(len(sorted_op_durations) * 0.7)]
        output.duration_p90 = sorted_op_durations[int(len(sorted_op_durations) * 0.9)]
        output.duration_p95 = sorted_op_durations[int(len(sorted_op_durations) * 0.95)]
        output.duration_max = sorted_op_durations[-1]
        output.total_duration = sum(sorted_op_durations)

    # proceed with sequential execution:
    unprocessed_seq_ops = deque(sequence_of_ops)
    unprocessed_seq_tensors = deque(unprocessed_tensors)

    global_timestamp = 0.0
    execution_step = 0
    while len(unprocessed_seq_ops) > 0 or len(unprocessed_seq_tensors) > 0:
        # process operators
        n_local_processed_ops = 0
        for op in unprocessed_seq_ops:
            depends_on = op._attrs["inputs"]

            # are all prereqs complete?
            can_proceed = all(
                tensor in output.tensor_sequential_trackers for tensor in depends_on
            )
            if can_proceed:
                # yes. This operator is ready to be executed.
                execution_step += 1

                op_duration = output.op_durations[op]

                output.op_sequential_trackers[op] = TimestampTracking(
                    execution_start=global_timestamp,
                    duration=op_duration,
                    execution_order=execution_step,
                )

                # modify global clock
                global_timestamp += op_duration

                n_local_processed_ops += 1
            else:
                # cannot go ahead, some tensors need to be marked as processed
                break

        for _ in range(0, n_local_processed_ops):
            unprocessed_seq_ops.popleft()

        # process tensors
        n_local_processed_tensors = 0
        for tensor in unprocessed_seq_tensors:
            depends_on = tensor.src_ops()

            # are all prereqs complete?
            can_proceed = all(op in output.op_sequential_trackers for op in depends_on)
            if can_proceed:
                # yes. The tensor computation is finished.
                max_execution_end = max(
                    output.op_sequential_trackers[op].execution_end for op in depends_on
                )
                max_execution_order = max(
                    output.op_sequential_trackers[op].execution_order
                    for op in depends_on
                )

                output.tensor_sequential_trackers[tensor] = TimestampTracking(
                    execution_start=max_execution_end,
                    duration=0.0,
                    execution_order=max_execution_order,
                )

                n_local_processed_tensors += 1
            else:
                # cannot proceed, some ops needs to be run first
                break

        for _ in range(0, n_local_processed_tensors):
            unprocessed_seq_tensors.popleft()

        # are we done?
        if n_local_processed_ops == 0 and n_local_processed_tensors == 0:
            # yes, no operators or tensors were processed on the current step.
            # This does not imply that all operators and tensors were processed.
            # Basically, this is a kinda early termination verification that
            # indicates that there is some invalid profiler / graph data.
            # So, we're trying to avoid infinite loops.
            break

    # process with parallel execution
    unprocessed_par_ops = set(sequence_of_ops)
    unprocessed_par_tensors = set(unprocessed_tensors)

    execution_step = 0
    while len(unprocessed_par_ops) > 0 or len(unprocessed_par_tensors) > 0:
        # process operators
        new_processed_ops: Set[Operator] = set()
        for op in unprocessed_par_ops:
            depends_on = op._attrs["inputs"]

            # are all prereqs complete?
            can_proceed = all(
                tensor in output.tensor_parallel_trackers for tensor in depends_on
            )
            if can_proceed:
                # yes. This operator is ready to be executed.
                op_duration = output.op_durations[op]

                max_execution_end = max(
                    output.tensor_parallel_trackers[tensor].execution_end
                    for tensor in depends_on
                )

                output.op_parallel_trackers[op] = TimestampTracking(
                    execution_start=max_execution_end,
                    duration=op_duration,
                    execution_order=execution_step,
                )

                new_processed_ops.add(op)

        # ok, there were some processed operators
        if len(new_processed_ops) > 0:
            for op in new_processed_ops:
                unprocessed_par_ops.remove(op)

            execution_step += 1

        # process tensors
        new_processed_tensors: Set[Tensor] = set()
        for tensor in unprocessed_par_tensors:
            depends_on = tensor.src_ops()

            # are all prereqs complete?
            can_proceed = all(op in output.op_parallel_trackers for op in depends_on)
            if can_proceed:
                # yes. The tensor computation is finished.
                max_execution_end = max(
                    output.op_parallel_trackers[op].execution_end for op in depends_on
                )
                max_execution_order = max(
                    output.op_parallel_trackers[op].execution_order for op in depends_on
                )

                output.tensor_parallel_trackers[tensor] = TimestampTracking(
                    execution_start=max_execution_end,
                    duration=0.0,
                    execution_order=max_execution_order,
                )

                new_processed_tensors.add(tensor)

        for tensor in new_processed_tensors:
            unprocessed_par_tensors.remove(tensor)

        # are we done?
        if len(new_processed_ops) == 0 and len(new_processed_tensors) == 0:
            # Same story: we're trying to avoid infinite loops.
            break

    # done
    return output
