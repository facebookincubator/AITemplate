# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from ..... import backend
from .....backend import registry
from .....utils import shape_utils
from ....base import Operator, Tensor

# pylint: disable=C0103,W0221,R1732,W0613
logging.basicConfig(level=logging.INFO)

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}KH = {{pooled_size}};
{{indent}}{{dtype}}KW = {{pooled_size}};
{{indent}}{{dtype}}NO = {{num_rois}};
{{indent}}{{dtype}}CO = CI;
{{indent}}{{dtype}}HO = {{pooled_size}};
{{indent}}{{dtype}}WO = {{pooled_size}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class roi_ops_base(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(
        self,
        num_rois,
        pooled_size,
        sampling_ratio,
        spatial_scale,
        position_sensitive,
        continuous_coordinate,
    ) -> None:
        """[summary]

        Parameters
        ----------
        stride : [type]
            [description]
        pad : [type]
            [description]
        dilate : int, optional
            [description], by default 1
        """
        super().__init__()
        self._attrs["op"] = "roi_align"
        self._attrs["num_rois"] = num_rois
        self._attrs["sampling_ratio"] = sampling_ratio
        self._attrs["spatial_scale"] = spatial_scale
        self._attrs["position_sensitive"] = position_sensitive
        self._attrs["continuous_coordinate"] = continuous_coordinate
        self._attrs["pooled_size"] = pooled_size
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int]):
        """[summary]

        Parameters
        ----------
        x : List[int]
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            num_rois=self._attrs["num_rois"],
            pooled_size=self._attrs["pooled_size"],
            position_sensitive=self._attrs["position_sensitive"],
        )

        output = {}
        exec(eval_func, output)  # noqa: P204  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor):
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key):
        """[summary]

        Parameters
        ----------
        key : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape):
        """[summary]

        Parameters
        ----------
        shape : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2], x_dim3=shape[3]
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        """
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        signature = "roi_align: num_rois=[{num_rois}], \
                                sampling_ratio=[{sampling_ratio}], \
                                spatial_scale=[{spatial_scale}], \
                                position_sensitive=[{position_sensitive}], \
                                continuous_coordinate=[{continuous_coordinate}], \
                                pooled_size=[{pooled_size}]".format(
            num_rois=self._attrs["num_rois"],
            sampling_ratio=self._attrs["sampling_ratio"],
            spatial_scale=self._attrs["spatial_scale"],
            position_sensitive=self._attrs["position_sensitive"],
            continuous_coordinate=self._attrs["continuous_coordinate"],
            pooled_size=self._attrs["pooled_size"],
        )
        return signature

    def __call__(self, x: Tensor, rois: Tensor) -> List[Tensor]:
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]

        Returns
        -------
        List[Tensor]
            [description]
        """
        self._attrs["inputs"] = [x, rois]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )

    def gen_function_decl(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs["name"])

    def gen_function_call(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs["name"])
