import ast
import builtins
import copy
import inspect
import logging
import textwrap
import warnings
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.jit as jit
import torch.nn as nn
from torch._sources import normalize_source_lines
from torch.fx import Graph, Tracer
from torch.fx.experimental.normalize import NormalizeArgs
from torch.fx.node import Argument, Node, Target

from . import acc_normalizer, acc_ops, acc_shape_prop, acc_utils  # noqa: F401


_LOGGER = logging.getLogger(__name__)


def _get_exception_wrapper_attr_name(exc_type: Type[Exception]) -> str:
    return f"_conditional_exception_wrapper_{exc_type.__name__}"


class Acc_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    def __init__(self):
        super().__init__()
        self.exceptions_rewritten: Set[Type[Exception]] = set()
        self.exceptions_bool_rewritten: Set[Type[Exception]] = set()

    def rewrite(
        self, fn: FunctionType
    ) -> Tuple[FunctionType, Set[Type[Exception]], Set[Type[Exception]]]:

        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = "".join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled function from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)  # noqa P204
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) <= 1
        fn_compiled = globals_dict[fn.__name__]

        # Return the correct FunctionType object and the Exceptions that were
        # rewritten during visit_If.
        return fn_compiled, self.exceptions_rewritten, self.exceptions_bool_rewritten

    def visit_Assert(self, node: ast.Assert):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse("torch._assert()", mode="eval")
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)

    def visit_If(self, if_node: ast.If):
        """
        Swap out the pattern `If(x): Raise(y)` with a ConditionalExceptionWrapper
        specialized for the specific exception y. The specialized
        ConditionalExceptionWrapper module will be added in the RewrittenModule.
        Only works with builtin Exceptions, as we assume the signature of the
        init for the Exception is a string.
        """
        raise_node = if_node.body[0]
        if not isinstance(raise_node, ast.Raise):
            return if_node

        # Don't handle orelse for now.
        # TODO: Move orelse to the body after calling ConditionalExceptionWrapper.
        if len(if_node.orelse) != 0:
            return if_node

        def _reuse_loc(node):
            return ast.copy_location(node, if_node)

        # If the exception has a message then we expect the raise's exc to be a
        # Call w/ a msg. Else if it's a exc Name then there's no msg to use.
        node_for_exc = raise_node.exc
        if isinstance(node_for_exc, ast.Name):
            # E.g. `raise AssertionError`, i.e. without an exc_msg.
            name_node_of_exc = node_for_exc
            exc_msg = _reuse_loc(ast.Constant(None))
        elif isinstance(node_for_exc, ast.Call):
            # E.g. `raise AssertionError("error message")`
            name_node_of_exc = node_for_exc.func  # type: ignore[assignment]
            if not isinstance(name_node_of_exc, ast.Name):
                return if_node
            # Most assertions just take a single string arg, but some may not; skip
            # handling such assertions for now.
            if len(node_for_exc.args) != 1:
                return if_node
            exc_msg = node_for_exc.args[0]
        else:
            return if_node

        # Convert what we expect is the name of the exception into its
        # associated python class.
        name_of_exc = name_node_of_exc.id
        try:
            exc_type = eval(name_of_exc)  # noqa P204
        except Exception:
            return if_node

        # Check that we actually have a builtin exception.
        if (
            not issubclass(exc_type, Exception)
            or getattr(getattr(exc_type, "__class__", None), "__module__", None)
            != "builtins"
        ):
            return if_node

        # We need a ConditionalExceptionWrapper specialized for every kind of
        # exception, so add it to exceptions_rewritten to remember for later to
        # add a specialized attr with it.
        self.exceptions_rewritten.add(exc_type)

        # From here we definitely should be able to do the replacement. Create a
        # Call node to the ConditionalExceptionWrapper module we're replacing
        # the If with, with args set as the If's condition and the string of the
        # exception. The call to the self._conditional_exception_wrapper_*Error
        # module is safe because the RewrittenModule will add it as an attr
        # based on the returned exceptions_rewritten, and we assume we are
        # currently modifying the AST of a method from a RewrittenModule.
        exc_wrapper_node = ast.parse(
            f"self.{_get_exception_wrapper_attr_name(exc_type)}()", mode="eval"
        )
        assert isinstance(exc_wrapper_node, ast.Expression)
        exc_wrapper_call_node = exc_wrapper_node.body
        assert isinstance(exc_wrapper_call_node, ast.Call)
        if isinstance(if_node.test, ast.BoolOp) and isinstance(
            if_node.test.op, ast.And
        ):
            self.exceptions_bool_rewritten.add(exc_type)
            bool_wrapper_node = ast.parse(
                f"self.{_get_exception_wrapper_attr_name(exc_type)}_bool()", mode="eval"
            )
            assert isinstance(exc_wrapper_node, ast.Expression)
            bool_wrapper_call_node = bool_wrapper_node.body
            assert isinstance(exc_wrapper_call_node, ast.Call)
            bool_wrapper_call_node.args = if_node.test.values
            exc_wrapper_call_node.args = [
                _reuse_loc(bool_wrapper_call_node),
                exc_msg,
            ]
        else:
            exc_wrapper_call_node.args = [if_node.test, exc_msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = _reuse_loc(ast.Expr(_reuse_loc(exc_wrapper_call_node)))

        # Return the new node to signify that we want to use it as a replacement
        # for the original `If x: Raise y` pattern.
        return expr_wrapper


class ConditionalExceptionWrapper(nn.Module):
    """
    This wrapper class is used to wrap conditional raising of exceptions during
    rewriting. For example:

    .. code-block:: python

        if self.name != "x":
            raise AssertionError(f"Name was not x: {self.name}")

    Is rewritten into

    .. code-block:: python

        self._conditional_exception_wrapper_AssertionError(
            self.name != "x", f"Name was not x: {self.name}"
        )

    Note that __init__ takes the Exception class that it is wrapping, while
    forward takes the condition to check and the message for the exception.

    """

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(self, exc: Type[Exception]):
        super().__init__()
        self.exc = exc

    def forward(self, cond: bool, msg: str):
        if cond:
            raise self.exc if msg is None else self.exc(msg)


class ConditionalExceptionBoolCondWrapper(nn.Module):
    """
    This is a wrapper class to for boolean ops used inside conditionals
    raising exceptions.
    This currently only handles binary input cases for the `and` operator
    at one level of depth
    For example:

    .. code-block:: python

    if self.name != "x" and self.name != "y":
        raise AssertionError(f"Name was not x: {self.name}")

    rewrites the `self.name != "x" and self.name != "y"` with
    a `_conditional_exception_wrapper_AssertionError_bool` as follows:

    .. code-block:: python

        self._conditional_exception_wrapper_AssertionError(
            self._conditional_exception_wrapper_AssertionError_bool(self.name != "x" and self.name != "y"), f"Name was not x: {self.name}"
        )
    """

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(self, op):
        super().__init__()

    def forward(self, *conds: Iterable):
        return all(conds)


# Custom tracer that traces to the functional level and rewrites asserts and
# exceptions.
class AccRewritingTracer(Tracer):
    # Add an explicit check for mutable operations, which break symbolic tracing.
    check_mutable_operations = True
    # Disble proxying buffers, which currently breaks some quantization code
    proxy_buffer_attributes = False

    # Note: Treat ConditionalExceptionWrapper as a leaf so that we don't
    # trace into it, because it contains control flow and raises an exception.
    DEFAULT_LEAF_MODULE_LIST = {
        ConditionalExceptionBoolCondWrapper,
        ConditionalExceptionWrapper,
        torch.nn.quantized.Linear,
        torch.nn.quantized.Conv2d,
        torch.nn.intrinsic.quantized.ConvReLU2d,
        jit.ScriptModule,
        jit.RecursiveScriptModule,
        torch.nn.modules.activation.MultiheadAttention,
    }

    def is_leaf_module(self, m: nn.Module, mod_qual_name: str) -> bool:
        return getattr(m, "_base_class_origin", type(m)) in self.leaf_module_list

    def trace(
        self,
        root: nn.Module,
        concrete_args: Optional[Dict[str, Any]] = None,
        ast_rewriter_allow_list: Optional[Set] = None,
        leaf_module_list: Optional[Set] = None,
    ) -> Tuple[Graph, nn.Module]:
        self.leaf_module_list = self.DEFAULT_LEAF_MODULE_LIST
        if leaf_module_list:
            self.leaf_module_list.update(leaf_module_list)
        rewritten = _rewrite(root, ast_rewriter_allow_list, self.leaf_module_list)
        return super().trace(rewritten, concrete_args), rewritten

    # override TraceBase's method
    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """

        ## Hacky way to decide inplace ops
        if type(target) != str:
            name_target = target.__name__
        else:
            name_target = target

        allow_list = ["and_", "or_"]  # python  operator.and_,  operator.or_
        if (
            name_target[-1] == "_"
            and name_target[0] != "_"
            and not (name_target in allow_list)
            and kind != "placeholder"
        ):
            raise RuntimeError(
                f"Tried to trace mutable operation {name_target}. FX only supports functional code"
            )

        return self.graph.create_node(kind, target, args, kwargs, name, type_expr)


# List of modules that need rewriting to be supported for tracing.
DEFAULT_REWRITE_ALLOW_LIST = {
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
}


def _rewrite(
    mod_to_rewrite: nn.Module,
    allow_list: Optional[Set] = None,
    leaf_module_list: Optional[Set] = None,
) -> nn.Module:
    if allow_list is None:
        allow_list = DEFAULT_REWRITE_ALLOW_LIST
    else:
        allow_list = allow_list.union(DEFAULT_REWRITE_ALLOW_LIST)

    if not leaf_module_list:
        leaf_module_list = set()

    # Rewrite this module's functions as well as all recursive modules'
    # functions that are attrs of this moodule. Return the new, rewritten module
    # hierarchy.
    def rewrite_module(m: nn.Module):
        if isinstance(m, jit.ScriptModule):
            # ScriptModule cannot be rewritten, so bypass it. The issue is it
            # requires explicitly calling its `__init__()`, calling
            # `nn.Module.__init__()` in the derived `RewrittenModule` is not
            # enough. And even if we init it we can't do much with it.
            return m

        # If m is an already-rewritten RewrittenModule, then use the original base class.
        base_class: Type[nn.Module] = getattr(m, "_base_class_origin", type(m))

        # Keep track of all the ConditionalExceptionWrappers that the
        # Acc_Rewriter calls into in this module so we can add them in init
        # below.
        all_added_wrappers: Set[Type[Exception]] = set()
        all_added_bool_wrappers: Set[Type[Exception]] = set()

        # Note: Make this a subclass of our base class.
        class RewrittenModule(base_class):  # type: ignore[valid-type, misc]
            # Keep track of the base_class so that symbolic tracing can
            # determine what kind of module this originally was later on.
            _base_class_origin = base_class
            # Add suffix to qualname so it's easier to debug the origin of this module.
            __qualname__ = f"{base_class.__qualname__}__AccRewrittenModule"

            # Write all of the non-dunder or special methods from base_class
            # into RewrittenModule.
            for method_name in dir(base_class):
                method = getattr(base_class, method_name, None)
                if method is None and method_name not in {"__doc__"}:
                    _LOGGER.warning(
                        f"{__qualname__} does not have attribute {method_name}"
                    )

                if builtins.type(method) is not FunctionType:
                    continue

                # Always skip rewriting dunder methods, as they haven't (yet) been
                # problematic, and modifying them has caused issues previously.
                if method_name.startswith("__") and method_name.endswith("__"):
                    continue

                # Only rewrite those Modules explicitly in the allow_list.
                assert allow_list is not None
                if base_class not in allow_list:
                    vars()[method_name] = method
                else:
                    (
                        vars()[method_name],
                        added_wrappers,
                        added_bool_wrappers,
                    ) = Acc_Rewriter().rewrite(method)
                    all_added_wrappers.update(added_wrappers)
                    all_added_bool_wrappers.update(added_bool_wrappers)

            def __init__(self, orig):
                nn.Module.__init__(self)

                # Iterate over all added exception wrappers and add
                # ConditionalExceptionWrapper attrs for each.
                for exc_type in all_added_wrappers:
                    wrapper_name = _get_exception_wrapper_attr_name(exc_type)
                    assert not hasattr(self, wrapper_name)
                    setattr(
                        self,
                        wrapper_name,
                        ConditionalExceptionWrapper(exc_type),
                    )

                for exc_type in all_added_bool_wrappers:
                    wrapper_name = f"{_get_exception_wrapper_attr_name(exc_type)}_bool"
                    assert not hasattr(self, wrapper_name)
                    setattr(
                        self,
                        wrapper_name,
                        ConditionalExceptionBoolCondWrapper(exc_type),
                    )

                # Recursively rewrite and copy all module attrs of this module.
                for k, v in orig.__dict__.items():
                    if k == "_modules":
                        for mod_k, mod_v in v.items():
                            if getattr(mod_v, "_base_class_origin", type(mod_v)) in leaf_module_list:  # type: ignore[operator]
                                _LOGGER.info(
                                    f"Skip rewriting leaf module {type(mod_v)}"
                                )
                                self._modules[mod_k] = mod_v
                            else:
                                self._modules[mod_k] = rewrite_module(mod_v)
                    else:
                        self.__dict__[k] = v

        # Add suffix to name so it's easier to debug the origin of this module.
        RewrittenModule.__name__ = f"{base_class.__name__}__AccRewrittenModule"
        return RewrittenModule(m)

    return rewrite_module(mod_to_rewrite)


def _remove_assertions(gm: torch.fx.GraphModule) -> bool:
    """
    Unconditionally removes all assertions found in GraphModule gm.
    Returns whether the graph is modified.
    """
    changed = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch._assert:
            gm.graph.erase_node(node)
            changed = True
    return changed


def _remove_exceptions(gm: torch.fx.GraphModule) -> bool:
    """
    Unconditionally removes all call_modules to ConditionalExceptionWrappers
    found in GraphModule gm. Returns whether the graph is modified.
    """
    changed = False
    for node in reversed(gm.graph.nodes):
        if node.op == "call_module" and (
            isinstance(gm.get_submodule(node.target), ConditionalExceptionWrapper)
            or isinstance(
                gm.get_submodule(node.target), ConditionalExceptionBoolCondWrapper
            )
        ):
            gm.graph.erase_node(node)
            changed = True
    return changed


def _replace_tensor_meta_with_rank(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op != "output" and "tensor_meta" in node.meta:
            node.meta["tensor_rank"] = acc_utils.map_tensor_metadata(
                node.meta["tensor_meta"], lambda x: len(x.shape)
            )
            del node.meta["tensor_meta"]


def rewriter_base_trace(mod, ast_rewriter_allow_list, leaf_module_list):
    rewritten_graph, rewritten_mod = AccRewritingTracer().trace(
        mod,
        ast_rewriter_allow_list=ast_rewriter_allow_list,
        leaf_module_list=leaf_module_list,
    )

    assert isinstance(rewritten_mod, nn.Module)
    # Note: use the rewritten_mod here as the root. This is necessary because
    # RewrittenModule includes a new module for the ConditionalExceptionWrapper.
    return torch.fx.GraphModule(rewritten_mod, rewritten_graph)


def trace(
    mod: nn.Module,
    sample_inputs: Sequence[Any],
    remove_assertions: bool = True,
    remove_exceptions: bool = True,
    use_acc_normalization: bool = True,
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None,
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None,
    acc_normalization_block_list: Optional[
        Set[Tuple[str, Union[str, Callable]]]
    ] = None,
) -> torch.fx.GraphModule:
    """
    Performs tracing and arg normalization specialized for accelerator lowering.

    It first rewrites the AST of the module's methods (and all attr methods
    recursively) to transform un-tracable parts of the module to make them
    traceable.

    It then traces to the functional level so that optimizations and backend
    accelerator importers have the ability to see and/or change inputs to each
    op.

    It then removes assertions and exception wrappers found during symbolic
    tracing if requested based on remove_assertions and remove_exceptions

    Dead code is then eliminated, which will e.g. remove any nodes that were
    only used by assertions or exceptions if they were removed.

    It then performs normalization on args/kwargs, aligning any arg that can be
    moved to kwarg to be so, and then making default values explicit.

    Args:

        mod (Module): The module to transform and trace.

        sample_inputs (Tuple[Union[torch.Tensor, List[torch.Tensor]]]):
                Sample inputs with which to run shape prop.

        remove_assertions (bool): Whether to remove assertion nodes from
                                    the graph after symbolic tracing.

        remove_exceptions (bool): Whether to remove exception wrapper nodes
                                    from the graph after symbolic tracing.

        use_acc_normalization (bool): Whether to use acc-specific
                                        normalization to all acc_ops.

        ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
                                            modules that need AST rewriting.

        leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
                                            modules will not be traced into.

        acc_normalization_block_list (Optional[Set[Tuple[str, Union[str, Callable]]]]):
                                    Optional set of (op, target) pairs to not apply acc
                                    normalization to. Just like the register_acc_op decarators,
                                    the target can either be a string (e.g. for op == "call_method")
                                    or a callable (e.g. for op == "call_function").
    """
    if mod.training:
        warnings.warn(
            "acc_tracer does not support currently support models for training."
            " Calling eval on model before tracing."
        )
        mod.eval()

    assert isinstance(sample_inputs, (list, tuple))

    # Rewrite the module to make it symbolic traceable, and then trace it.
    traced = rewriter_base_trace(mod, ast_rewriter_allow_list, leaf_module_list)

    # Now remove all assertions and exceptions if requested.
    if remove_assertions:
        _remove_assertions(traced)
    if remove_exceptions:
        _remove_exceptions(traced)

    # Cleanup any dead code from the original module as well as resulting dead
    # nodes after removing assertions and exceptions.
    traced.graph.eliminate_dead_code()
    traced.recompile()

    # Run shape prop to add node.meta["type"] to nodes, needed for NormalizeArgs.
    acc_shape_prop.AccShapeProp(traced).propagate(*sample_inputs)
    # Swap out tensor_meta for tensor_rank, because we don't actually want to rely on
    # tensor_meta yet for normalization/lowering, though rank shouldn't change.
    _replace_tensor_meta_with_rank(traced)
    # Now normalize args/kwargs to make default values visible. Leave args/kwargs as
    # they were, since all-kwarg normalization is broken, and we don't need it anyway.
    traced = NormalizeArgs(traced, normalize_to_only_use_kwargs=False).transform()

    # Normalize to acc-specialized wrappers for consistency across op naming and
    # ensuring all kwarg usage.
    if use_acc_normalization:
        acc_normalizer.normalize(
            traced, acc_normalization_block_list=acc_normalization_block_list
        )

    traced.recompile()

    # Run shape prop to again to populate tensor_meta after normalize.
    acc_shape_prop.AccShapeProp(traced).propagate(*sample_inputs)

    return traced
