# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Registry is a design pattern to map a string key to a function.
The registry decorator is mainly used for backend functions.
"""

from __future__ import annotations

from typing import Callable

BACKEND_FUNCTIONS = {}


def reg(func_name: str, func: Callable = None) -> Callable:
    """Register a new function

    Example

    .. highlight:: python
    .. code-block:: python

        @registry.reg("func_name")
        def func(args):
            ....


    Parameters
    ----------
    func_name : str
        Registry key for the function
    func : Callable, optional
        Function to be registered, by default None

    Returns
    -------
    Callable
        Function in registry

    Raises
    ------
    RuntimeError
        If same key is founded in registry, will raise a RuntimeError
    """
    if func_name in BACKEND_FUNCTIONS:
        raise RuntimeError("{name} funcion has been registered.".format(name=func_name))

    def _do_reg(func):
        BACKEND_FUNCTIONS[func_name] = func
        return func

    if func is None:
        return _do_reg
    return func


def get(func_name: str) -> Callable:
    """Get a function from registry by using a key

    Example

    .. highlight:: python
    .. code-block:: python

        func = registry.get("func_name")
        func(args)



    Parameters
    ----------
    func_name : str
        Key for function in registry

    Returns
    -------
    Callable
        Function associated with the key

    Raises
    ------
    RuntimeError
        If key is not founded in registry, will raise a RuntimeError
    """
    if func_name not in BACKEND_FUNCTIONS:
        raise RuntimeError(f"{func_name} function has not been registered.")
    return BACKEND_FUNCTIONS[func_name]
