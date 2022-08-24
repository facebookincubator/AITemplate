# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""

from .... import backend
from ....backend import registry
from ...base import Operator

# pylint: disable=C0103,W0221


class slice_scatter(Operator):
    """This op represents a special fusion case where the
    inputs of a concatenate op all come from slice ops. In such a case,
    we can remove the concatenate op by placing each slice's output
    into the correct location in the original concatenate's output.
    """

    @staticmethod
    def is_valid(cat_op: Operator) -> bool:
        if cat_op._attrs["op"] != "concatenate":
            return False
        return all(
            x._attrs["src_ops"] is not None
            and len(x._attrs["src_ops"]) == 1
            and len(x._attrs["dst_ops"]) == 1
            and list(x._attrs["src_ops"])[0]._attrs["op"] == "dynamic_slice"
            for x in cat_op._attrs["inputs"]
        )

    def _update_inputs_outputs(self, cat_op):
        self._attrs["inputs"] = []
        for slice_op in self._attrs["slice_ops"]:
            assert (
                len(slice_op._attrs["inputs"]) == 1
            ), "Slice op should only have 1 input! op: {}".format(slice_op)
            input_tensor = slice_op._attrs["inputs"][0]
            # A slice op's output may be fed into the same cat op multiple
            # times, so we make sure it's removed from the set only once.
            if slice_op in input_tensor._attrs["dst_ops"]:
                input_tensor._attrs["dst_ops"].remove(slice_op)
                input_tensor._attrs["dst_ops"].add(self)
            self._attrs["inputs"].append(input_tensor)

        self._attrs["outputs"] = cat_op._attrs["outputs"]
        for y in self._attrs["outputs"]:
            y._attrs["src_ops"] = {self}

        for op in self._attrs["slice_ops"]:
            op._attrs["outputs"][0]._attrs["src_ops"] = set()
            op._attrs["outputs"][0]._attrs["dst_ops"] = set()

        for x in cat_op._attrs["inputs"]:
            x._attrs["src_ops"] = set()
            x._attrs["dst_ops"] = set()

    def __init__(self, cat_op: Operator) -> None:
        """[summary]

        Parameters
        ----------

        Returns
        """
        super().__init__()
        assert slice_scatter.is_valid(cat_op)

        self._attrs["op"] = "slice_scatter"
        self._attrs["has_profiler"] = False
        self._attrs["scatter_dim"] = cat_op._attrs["concat_dim"]
        slice_ops = []
        for x in cat_op._attrs["inputs"]:
            src_ops = x.src_ops()
            assert len(src_ops) == 1
            slice_op = list(src_ops)[0]
            slice_ops.append(slice_op)
        self._attrs["slice_ops"] = slice_ops

        self._update_inputs_outputs(cat_op)
        self._set_depth()

    def __call__(self):
        """[summary]

        Parameters
        ----------

        Returns
        -------
        """
        raise RuntimeError("op {} cannot be called directly".format(self._attrs["op"]))

    def _get_func(self, fmt_str):
        """[summary]

        Parameters
        ----------
        inputs : string
            [description] format string to create func_key for looking up func
                          from the registry

        Returns
        -------
        [type]
            [description]
        """
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_decl")
        return func(self._attrs)

    def gen_function_call(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_call")
        return func(self._attrs)
