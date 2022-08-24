# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from collections import OrderedDict

from .parameter import Parameter

# pylint: disable=W0221,R1705,W0235


class Module(object):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self):
        """[summary]"""
        self._children = OrderedDict()
        self._reg_params = OrderedDict()

    def __setattr__(self, name, value):
        """[summary]

        Parameters
        ----------
        name : [type]
            [description]
        value : [type]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        if hasattr(self, name):
            raise RuntimeError("%s has been registed in the block" % name)
        if isinstance(value, Module):
            self._children[name] = value
        elif isinstance(value, Parameter):
            self._reg_params[name] = value

        super().__setattr__(name, value)

    def _check_container_with_block(self):
        """[summary]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        children = set(self._children.values())

        def _find_unregistered_block_in_container(data):
            if isinstance(data, (list, tuple)):
                for ele in data:
                    if _find_unregistered_block_in_container(ele):
                        return True
                return False
            elif isinstance(data, dict):
                for _, v in data.items():
                    if _find_unregistered_block_in_container(v):
                        return True
                    return False
            elif isinstance(data, Module):
                return data not in (c for c in children)
            return False

        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple, dict)) and not (
                k.startswith("__") or k == "_children"
            ):
                if _find_unregistered_block_in_container(v):
                    raise RuntimeError(
                        "Blocks inside the list, tuple or dict will not be registered automatically"
                    )

    def _register_child(self, child, name=None):
        """[summary]

        Parameters
        ----------
        child : [type]
            [description]
        """
        if name is None:
            name = str(hash(child))
        self._children[name] = child

    def parameters(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        self._check_container_with_block()
        tmp = []
        ret = []
        tmp.extend(self._reg_params.values())
        for _, child in self._children.items():
            tmp.extend(child.parameters())
        for param in tmp:
            tensor = param.tensor()
            if tensor._attrs["is_param"]:
                ret.append(param)
        if len(ret) == 0:
            # if forget to run mark_tensor_param pass, return as is
            return tmp
        return ret

    def __call__(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        out = self.forward(*args)
        return out

    def forward(self, *args):
        """[summary]

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError


class Sequential(Module):
    """[summary]

    Parameters
    ----------
    Block : [type]
        [description]
    """

    def __init__(self):
        """[summary]"""
        super().__init__()

    def add_module(self, name, module):
        """[summary]"""
        self._register_child(module, name)

    def forward(self, x, *args):
        """[summary]

        Parameters
        ----------
        x : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        for block in self._children.values():
            x = block(x, *args)
            args = []
            if isinstance(x, (tuple, list)):
                args = x[1:]
                x = x[0]
        if args:
            x = tuple([x] + list(args))
        return x
