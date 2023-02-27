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
import logging
import operator
from typing import Any, NamedTuple

import torch
import torch.fx
from fx2ait.tools.ait_subgraph_rewriter import replace_pattern

from torch.fx.experimental.const_fold import split_const_subgraphs
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import TensorMetadata

_LOGGER = logging.getLogger(__name__)

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any

from fx2ait.acc_tracer import acc_ops
from torch.fx import symbolic_trace


def replacement_pattern_abstract(replacement):
    """
    Replace the pattern graph by a node of call_function of this `replacement`
    """
    traced = symbolic_trace(replacement)
    replacement_placeholders = [
        node for node in traced.graph.nodes if node.op == "placeholder"
    ]
    for n in traced.graph.nodes:
        if n.op == "output":
            before_output = n.all_input_nodes[0]
            with traced.graph.inserting_after(before_output):
                new_args = tuple(replacement_placeholders)
                new_node = traced.graph.create_node(
                    "call_function",
                    replacement,
                    args=new_args,
                    kwargs=None,
                )
                before_output.replace_all_uses_with(new_node)
    traced.graph.eliminate_dead_code()
    traced.recompile()
    return traced


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module.
    def skip_folding(node: torch.fx.Node):
        if node.target == torch.ops.aten.sym_size:
            return True

    const_split_mod = split_const_subgraphs(
        traced_mod, skip_folding_node_fn=skip_folding
    )
    const_split_mod.run_folding()
    return const_split_mod


def nchw2nhwc_pass(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    This pass is a kind of hacky way to support some vision models. The reason is due the fact that the frontend is traced based on channel first while AIT needs channel last.
    We need to modify
    1) mean.dim for dim=[-1,-2] changed to [-2,-3]
    2) dim=1 of mean.dim changed to dim=3
    3) concat(inputs, dim=1) need to be dim=3
    """
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.cat.default:
            if n.args[1] == 1:
                new_args = list(n.args)
                new_args[1] = 3
        elif n.op == "call_function" and n.target == torch.ops.aten.mean.dim:
            if n.args[1] == [-1, -2] or [-2, -1]:
                new_args = list(n.args)
                new_args[1] = [-2, -3]
        else:
            continue
        n.args = tuple(new_args)
        modified = True

        modified_list1 = []
        modified_list2 = []
        modified_list3 = []
        for u in n.users:
            if u.target == torch.ops.aten.sym_size and u.args[1] == 1:
                modified_list1.append(u)
            if u.target == torch.ops.aten.sym_size and u.args[1] == 2:
                modified_list2.append(u)
            if u.target == torch.ops.aten.sym_size and u.args[1] == 3:
                modified_list3.append(u)

        for v in modified_list1:
            new_args = list(v.args)
            new_args[1] = 3
            v.args = tuple(new_args)

        for v in modified_list2:
            new_args = list(v.args)
            new_args[1] = 1
            v.args = tuple(new_args)

        for v in modified_list3:
            new_args = list(v.args)
            new_args[1] = 2
            v.args = tuple(new_args)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


# TODO: delete in future
def replace_inplace_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    Remove this func after functionalization is workable
    """
    modified = False
    map_func = {
        torch.ops.aten.relu_.default: torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh_.default: torch.ops.aten.hardtanh.default,
        torch.ops.aten.add_.Tensor: torch.ops.aten.add.Tensor,
    }
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in map_func.keys():
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_args = node.args
                new_node = module.graph.create_node(
                    "call_function",
                    map_func[node.target],
                    args=new_args,
                    kwargs=None,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_native_layernorm_with_layernorm(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if (
            n.op == "call_function"
            and n.target == torch.ops.aten.native_layer_norm.default
        ):
            for v in n.users:
                if v.op == "call_function" and v.target == operator.getitem:
                    if v.args[1] != 0:
                        raise RuntimeError(
                            f"Got args[{v.args[1]}]!!\n"
                            "layernorm can only generate output (args[0]), "
                            "not mean (args[1]) or std (args[2])!"
                        )
                    new_op = torch.ops.aten.layer_norm.default
                    new_args = (*n.args, True)  # cudnn_enable=True
                    modified = True
                else:
                    continue

                with module.graph.inserting_after(v):
                    new_node = module.graph.create_node(
                        "call_function",
                        new_op,
                        args=new_args,
                        kwargs=v.kwargs,
                    )
                    v.replace_all_uses_with(new_node)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_transpose_mm_op_with_linear(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.t.default:
            to_erase = []
            for v in n.users:
                if v.op == "call_function" and v.target == torch.ops.aten.addmm.default:
                    new_op = torch.ops.aten.linear
                    bias, inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, bias)
                    modified = True
                elif v.op == "call_function" and v.target == torch.ops.aten.mm.default:
                    new_op = torch.ops.aten.linear
                    inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, None)
                    modified = True
                # this pass should be after `compose_bmm`
                elif v.op == "call_function" and v.target == aten_compose_bmm_2d:
                    new_op = torch.ops.aten.linear
                    inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, None)
                    modified = True
                else:
                    continue

                with module.graph.inserting_after(v):
                    new_node = module.graph.create_node(
                        "call_function",
                        new_op,
                        args=new_args,
                        kwargs=v.kwargs,
                    )
                    v.replace_all_uses_with(new_node)
                    to_erase.append(v)
            for v in to_erase:
                module.graph.erase_node(v)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_batch_norm(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Current exir.capture enable_aot and it captures bwd needed nodes in output P619801318
    This pass removes those unused node and replace with classic aten.batch_norm
    """
    batch_node_list = []
    for n in module.graph.nodes:
        if n.target == torch.ops.aten._native_batch_norm_legit_functional.default:
            batch_node_list.append(n)
        if n.target == "output":
            output_node = n

    if len(batch_node_list) > 0:
        modified = True
    else:
        modified = False
    for n in batch_node_list:
        new_op = torch.ops.aten.batch_norm
        new_args = list(n.args)
        new_args.append(False)
        new_args = tuple(new_args)
        user_list = [x for x in n.users]
        user_list_copy_node = []
        user_list_copy_node.append(next(iter(user_list[1].users)))
        user_list_copy_node.append(next(iter(user_list[2].users)))
        getitem_node = user_list[0]
        with module.graph.inserting_after(getitem_node):
            new_node = module.graph.create_node(
                "call_function",
                new_op,
                args=new_args,
                kwargs=n.kwargs,
            )
            getitem_node.replace_all_uses_with(new_node)

        output_args = output_node.args[0]
        new_output_args = [x for x in output_args if x not in user_list_copy_node]
        output_node.args = (new_output_args,)
        module.graph.eliminate_dead_code()
        module.recompile()
    return PassResult(module, modified)


def replace_aten_op_with_indices(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.native_batch_norm.default,
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten._native_batch_norm_legit_no_training.default,
        ):
            modified = True
            if len(n.users) != 1:
                raise RuntimeError(
                    f"{n.target} has users={len(n.users)}. We can only handle it with 1 user"
                )
            if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                new_op = torch.ops.aten.max_pool2d
                new_args = n.args
            elif n.target == torch.ops.aten.max_pool3d_with_indices.default:
                new_op = torch.ops.aten.max_pool3d
                new_args = n.args
            elif (
                n.target == torch.ops.aten.native_batch_norm.default
                or n.target == torch.ops.aten._native_batch_norm_legit.default
            ):
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                new_args = tuple(new_args)
            elif (
                n.target == torch.ops.aten._native_batch_norm_legit_no_training.default
            ):
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                # _native_batch_norm_legit_no_training doesn't take in a training arg (assumed to be false)
                # but batchnorm takes in a training arg at position 5.
                new_args.insert(5, False)
                new_args = tuple(new_args)

            getitem_node = next(iter(n.users))
            with module.graph.inserting_after(getitem_node):
                new_node = module.graph.create_node(
                    "call_function",
                    new_op,
                    args=new_args,
                    kwargs=n.kwargs,
                )
                getitem_node.replace_all_uses_with(new_node)
                module.graph.erase_node(getitem_node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_aten_reshape_alias_with_replace(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    # The stride parameter is not used. Replace with reshape without stride
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten._reshape_alias.default,
        ):
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_args = (node.args[0], node.args[1])
                new_node = module.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=new_args,
                    kwargs=None,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
            break
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


## Acc tracer pass, but for aten usage
def acc_replace_reshape_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    Change TensorMetadata to shapeMetadata which only contains shape field.
    """
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == acc_ops.reshape:
            if isinstance(n.kwargs["acc_out_ty"], TensorMetadata):

                class shapeMetadata(NamedTuple):
                    shape: torch.Size

                node = n
                with module.graph.inserting_after(node):
                    new_kargs = {}
                    new_kargs["input"] = node.kwargs["input"]
                    new_kargs["acc_out_ty"] = shapeMetadata(
                        node.kwargs["acc_out_ty"].shape
                    )
                    new_node = module.graph.create_node(
                        "call_function",
                        acc_ops.reshape,
                        args=node.args,
                        kwargs=new_kargs,
                    )
                    node.replace_all_uses_with(new_node)
                    module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return module


def remove_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    1. Remove clone, _unsafe_view node. #TODO Remove this func after functionalization is workable
    2. Remove inefficient op getitem(index=slice) P561572458
    """
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (torch.ops.aten.clone.default,):
            modified = True
            node = n
            input_n = node.all_input_nodes[0]
            node.replace_all_uses_with(input_n)
    module.graph.eliminate_dead_code()
    module.recompile()
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten._unsafe_view.default,
        ):
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_node = module.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=node.args,
                    kwargs=node.kwargs,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def aten_operator_getitem(*args):
    return operator.getitem(*args)


def replace_builtin_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    To differential the same op in fx2ait as they are registered in the same dictionary
    """

    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (operator.getitem,):
            modified = True
            n.target = aten_operator_getitem
    module.graph.eliminate_dead_code()
    module.recompile()

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


###############
"""
Trace compose. For some ops, we do not want to decompose further but want coarse granularity
For ex:
1. bmm
2. chunk
3. getitem(input, idx=(slice(),slice()...))
"""


def aten_compose_getitem_slice(input, list_args):
    for _, args in enumerate(list_args):
        input = torch.ops.aten.slice.Tensor(input, *args)
    return input


def compose_getitem_slice(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed getitem(input, idx=(slice(),slice()...))
    """

    def match_pattern(module, node):
        if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor:
            holder = []
            holder.append(node)
            qualified = True
            user_change_input = []

            while qualified:
                next_user = None
                for user in node.users:
                    if (
                        user.target == torch.ops.aten.slice.Tensor
                        and node.args[1] + 1 == user.args[1]
                    ):
                        next_user = user
                    elif (
                        user.target == torch.ops.aten.sym_size
                        and user.args[1] == node.args[1]
                    ):
                        user_change_input.append(user)
                    else:
                        qualified = False
                        break
                if qualified and next_user:
                    node = next_user
                    holder.append(node)
                else:
                    qualified = False

            if len(holder) == 1:
                return (False,)
            else:
                return (True, holder, user_change_input)
        return (False,)

    modified = False
    for node in module.graph.nodes:
        res = match_pattern(module, node)
        if res[0]:
            modified = True
            holder = res[1]
            user_change_input = res[2]
            input_n = holder[0].args[0]
            last_n = holder[-1]
            list_args = []
            for h_n in holder:
                list_args.append(h_n.args[1:])

            with module.graph.inserting_after(last_n):
                new_args = (input_n, list_args)
                new_node = module.graph.create_node(
                    "call_function",
                    aten_compose_getitem_slice,
                    args=tuple(new_args),
                    kwargs=None,
                )
            last_n.replace_all_uses_with(new_node)
            for n in user_change_input:
                new_args = list(n.args)
                new_args[0] = new_node
                n.args = tuple(new_args)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def aten_compose_mm_2d(arg0_1, arg1_1):
    sym_size = torch.ops.aten.sym_size(arg0_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(arg0_1, 1)
    mul = sym_size * sym_size_1
    sym_size_2 = torch.ops.aten.sym_size(arg0_1, 2)
    view = torch.ops.aten.view.default(arg0_1, [mul, sym_size_2])
    mm = torch.ops.aten.mm.default(view, arg1_1)
    sym_size_3 = torch.ops.aten.sym_size(arg1_1, 1)
    view_1 = torch.ops.aten.view.default(mm, [sym_size, sym_size_1, sym_size_3])
    return view_1


def aten_compose_bmm_2d(flat_args_1, flat_args_2):
    sym_size = torch.ops.aten.sym_size(flat_args_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(flat_args_1, 1)
    sym_size_2 = torch.ops.aten.sym_size(flat_args_1, 2)
    expand = torch.ops.aten.expand.default(
        flat_args_1, [sym_size, sym_size_1, sym_size_2]
    )
    view = torch.ops.aten.view.default(expand, [sym_size, sym_size_1, sym_size_2])
    sym_size_3 = torch.ops.aten.sym_size(flat_args_2, 0)
    sym_size_4 = torch.ops.aten.sym_size(flat_args_2, 1)
    expand_1 = torch.ops.aten.expand.default(
        flat_args_2, [sym_size, sym_size_3, sym_size_4]
    )
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size, sym_size_3, sym_size_4])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.view.default(bmm, [sym_size, sym_size_1, sym_size_4])
    return view_2


def aten_compose_bmm_3d(flat_args_1, flat_args_2):
    sym_size = torch.ops.aten.sym_size(flat_args_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(flat_args_1, 1)
    sym_size_2 = torch.ops.aten.sym_size(flat_args_1, 2)
    expand = torch.ops.aten.expand.default(
        flat_args_1, [sym_size, sym_size_1, sym_size_2]
    )
    view = torch.ops.aten.view.default(expand, [sym_size, sym_size_1, sym_size_2])
    sym_size_3 = torch.ops.aten.sym_size(flat_args_2, 1)
    sym_size_4 = torch.ops.aten.sym_size(flat_args_2, 2)
    expand_1 = torch.ops.aten.expand.default(
        flat_args_2, [sym_size, sym_size_3, sym_size_4]
    )
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size, sym_size_3, sym_size_4])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.view.default(bmm, [sym_size, sym_size_1, sym_size_4])
    return view_2


def compose_bmm(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed bmm (matmul)
    """
    modified = False
    # pattern replacement for aten_compose_mm_2d
    _LOGGER.info("compose_bmm: pattern matching for aten_compose_mm_2d...")
    aten_compose_mm_2d_replacement = replacement_pattern_abstract(aten_compose_mm_2d)
    res = replace_pattern(module, aten_compose_mm_2d, aten_compose_mm_2d_replacement)
    if len(res) > 0:
        modified = True
    # pattern replacement for aten_compose_bmm_2d
    _LOGGER.info("compose_bmm: pattern matching for aten_compose_bmm_3d...")

    def match_filter_aten_compose_bmm_2d(match, original_graph, pattern_graph):
        if len(match.placeholder_nodes[1].meta["val"].shape) == 2:
            return True
        else:
            return False

    aten_compose_bmm_2d_replacement = replacement_pattern_abstract(aten_compose_bmm_2d)
    res = replace_pattern(
        module,
        aten_compose_bmm_2d,
        aten_compose_bmm_2d_replacement,
        [match_filter_aten_compose_bmm_2d],
    )
    if len(res) > 0:
        modified = True
    # pattern replacement for aten_compose_bmm_3d
    _LOGGER.info("compose_bmm: pattern matching for aten_compose_bmm_2d...")

    def match_filter_aten_compose_bmm_3d(match, original_graph, pattern_graph):
        if len(match.placeholder_nodes[1].meta["val"].shape) == 3:
            return True
        else:
            return False

    aten_compose_bmm_3d_replacement = replacement_pattern_abstract(aten_compose_bmm_3d)
    res = replace_pattern(
        module,
        aten_compose_bmm_3d,
        aten_compose_bmm_3d_replacement,
        [match_filter_aten_compose_bmm_3d],
    )
    if len(res) > 0:
        modified = True

    return PassResult(module, modified)


def aten_compose_chunk(flat_args_1, chunk, dim):
    sym_size = torch.ops.aten.sym_size(flat_args_1, dim)
    add = operator.add(sym_size, chunk)
    sub = operator.sub(add, 1)
    floordiv = operator.floordiv(sub, chunk)
    split = torch.ops.aten.split.Tensor(flat_args_1, floordiv, dim)
    return split


def compose_chunk(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed chunk
    """

    def match_pattern(module, node):
        if node.op == "call_function" and node.target in (torch.ops.aten.split.Tensor,):
            div = node.args[1]
            input = node.args[0]
            if isinstance(div, int):
                return (False,)
            if div.target != operator.floordiv:
                return (False,)
            else:
                div_const = div.args[1]
                sub = div.args[0]
                if sub.target != operator.sub:
                    return (False,)
                else:
                    add = sub.args[0]
                    if add.target != operator.add:
                        return (False,)
                    else:
                        add_const = add.args[1]
                        if add_const != div_const:
                            return (False,)
                        symsize = add.args[0]
                        if symsize.target != torch.ops.aten.sym_size:
                            return (False,)
                        else:
                            symsize_input = symsize.args[0]
                            dim = symsize.args[1]
                            if symsize_input != input:
                                return (False,)

            return (True, div_const, dim)
        else:
            return (False,)

    modified = False
    for node in module.graph.nodes:
        res = match_pattern(module, node)
        if res[0]:
            modified = True
            with module.graph.inserting_after(node):
                new_args = (node.args[0], res[1], res[2])
                new_node = module.graph.create_node(
                    "call_function",
                    aten_compose_chunk,
                    args=new_args,
                    kwargs=None,
                )
            node.replace_all_uses_with(new_node)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


## TODO: we will remove this pass once dynamo fixed the bug
def acc_replace_mul_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    Put constant at the end of multiplicaiton, i.e change 15*x.size(1) to x.size(1)*15.
    """
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == acc_ops.mul:
            if isinstance(n.kwargs["input"], int):
                node = n
                with module.graph.inserting_after(node):
                    new_kargs = {}
                    new_kargs["input"] = node.kwargs["other"]
                    new_kargs["other"] = node.kwargs["input"]
                    new_node = module.graph.create_node(
                        "call_function",
                        acc_ops.mul,
                        args=node.args,
                        kwargs=new_kargs,
                    )
                    node.replace_all_uses_with(new_node)
                    module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return module
