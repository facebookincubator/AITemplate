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
import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union

import torch

from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

logger: logging.Logger = logging.getLogger(__name__)


class AITSubgraphMatcher(SubgraphMatcher):
    def __init__(
        self,
        pattern: Graph,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
    ):
        super(AITSubgraphMatcher, self).__init__(
            pattern, match_output, match_placeholder, remove_overlapping_matches
        )

    def _match_args(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
        logger.info(f"  matching arguments: {pn} to {gn}")
        assert not (
            isinstance(pn, Node) and isinstance(gn, Node)
        ), "pn and gn cannot both be Node"
        if isinstance(pn, Node) and not isinstance(gn, Node):
            if pn.op == "placeholder":
                # Check if we've already matched these nodes in the current
                # traversal
                if pn in match.nodes_map:
                    return match.nodes_map[pn] == gn

                match.nodes_map[pn] = gn
                return True
            elif pn.op == "call_function" and pn.target == torch.ops.aten.sym_size:
                return True
            else:
                return False
        elif not isinstance(pn, Node) and isinstance(gn, Node):
            return False
        else:
            return type(gn) == type(pn) and gn == pn

    def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool:
        logger.info(f"  matching node: {pn} to {gn}")
        # breakpoint()
        assert isinstance(pn, Node) and isinstance(gn, Node), str(
            f"pn and gn must be Node, pn: {pn}, gn: {gn}"
        )
        if (
            pn.target == torch.ops.aten.sym_size
            and gn.target == torch.ops.aten.sym_size
        ):
            return True
        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficienty way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`, and save a local copy of match
        saved_match = copy.copy(match)
        match.nodes_map[pn] = gn

        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = True

        def flatten_args(args) -> List[Any]:
            # Recursively flatten args
            result: List[Any] = []
            for arg in args:
                # flatten the list, if only it's a list/tuple of nodes
                if (
                    isinstance(arg, (list, tuple))
                    and len(arg) > 0
                    and isinstance(arg[0], Node)
                ):
                    result.extend(flatten_args(arg))
                else:
                    result.append(arg)

            return result

        pn_flatten_args = flatten_args(pn.args)
        gn_flatten_args = flatten_args(gn.args)

        if pn.kwargs.keys() == gn.kwargs.keys():
            for key in pn.kwargs.keys():
                pn_flatten_args.append(pn.kwargs[key])
                gn_flatten_args.append(gn.kwargs[key])
        else:
            match_found = False

        if match_found and len(pn_flatten_args) == len(gn_flatten_args):
            for pn_, gn_ in zip(pn_flatten_args, gn_flatten_args):
                if isinstance(gn_, Node) and isinstance(pn_, Node):
                    matched = self._match_nodes(pn_, gn_, match)
                else:
                    matched = self._match_args(pn_, gn_, match)
                if not matched:
                    match_found = False
                    break
        else:
            match_found = False

        if not match_found:
            # revert to saved_match before matching with current node
            match = copy.copy(saved_match)
            return False

        return True


__all__ = [
    "Match",
    "replace_pattern",
    "ReplacedPatterns",
]


class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]


@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: List[Node]


def _replace_submodules(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_submodule(
        mod: torch.nn.Module, target: str
    ) -> Optional[torch.nn.Module]:
        try:
            mod_match = mod.get_submodule(target)
            return mod_match
        except AttributeError:
            return None

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":
            gm_submod = try_get_submodule(gm, node.target)

            replacement_submod = try_get_submodule(replacement, node.target)

            # CASE 1: This target already exists as a submodule in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_submod is not None:
                continue

            # CASE 2: The target exists as a submodule in `replacement`
            # only, so we need to copy it over.
            elif replacement_submod is not None:
                new_submod = copy.deepcopy(getattr(replacement, node.target))
                gm.add_submodule(node.target, new_submod)

            # CASE 3: The target doesn't exist as a submodule in `gm`
            # or `replacement`
            else:
                raise RuntimeError(
                    'Attempted to create a "',
                    node.op,
                    '" node during subgraph rewriting '
                    f"with target {node.target}, but "
                    "the referenced submodule does not "
                    "exist in either the original "
                    "GraphModule `gm` or the replacement"
                    " GraphModule `replacement`",
                )

    gm.graph.lint()


def replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule],
    match_filters: List[Callable[["InternalMatch", Graph, Graph], bool]] = None,  # type: ignore[name-defined]
) -> List[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2
    """
    match_and_replacements = _replace_pattern(gm, pattern, replacement, match_filters)
    return [
        Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements
    ]


def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule],
    match_filters: List[Callable[["InternalMatch", Graph, Graph], bool]] = None,  # type: ignore[name-defined]
) -> List[ReplacedPatterns]:

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    else:
        pattern_graph = symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    else:
        replacement_graph = symbolic_trace(replacement).graph
    matcher = AITSubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
    )

    _matches: List[InternalMatch] = matcher.match(original_graph)
    logger.info(f"matches = {_matches}")
    # Filter out matches that don't match the filter
    _matches = [
        m
        for m in _matches
        if all(
            match_filter(m, original_graph, pattern_graph)
            for match_filter in match_filters
        )
    ]

    replacement_placeholders = [
        n for n in replacement_graph.nodes if n.op == "placeholder"
    ]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = list(user_nodes)[0]
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(
                replacement_graph, val_map
            )

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes = []

        def get_replacement_nodes(curr_node: Node):
            nonlocal replacement_nodes
            for arg in curr_node.args:
                if isinstance(arg, Node):
                    if arg not in val_map.values():
                        get_replacement_nodes(arg)
            replacement_nodes.append(curr_node)

        for ret_node in copied_returning_nodes:
            get_replacement_nodes(ret_node)

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        logger.info(f"Remove pattern node from original graph, match={match}")
        for node in reversed(pattern_graph.nodes):
            if (
                node.op != "placeholder"
                and node.op != "output"
                and node.target != torch.ops.aten.sym_size
            ):

                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)
        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes,
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.graph.eliminate_dead_code()
    gm.recompile()
    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    # if isinstance(replacement, torch.nn.Module):
    #     _replace_submodules(gm, replacement)

    return match_and_replacements
