# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] roi_ops op
"""

from typing import List

from ....base import Tensor
from .roi_ops import roi_ops_base

# pylint: disable=C0103


class multi_level_roi_align(roi_ops_base):
    """[summary]

    Parameters
    ----------
    roi_ops_base : [type]
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
        im_shape,
    ) -> None:
        """[summary]

        Parameters
        ----------
        pooled_size : [type]
            [description]
        stride : [type]
            [description]
        pad : [type]
            [description]
        """
        super().__init__(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
        )
        self._attrs["op"] = "multi_level_roi_align"
        self._attrs["im_shape"] = im_shape

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
            num_rois=self._attrs["num_rois"] * x[0],
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

    def __call__(
        self, p2: Tensor, p3: Tensor, p4: Tensor, p5: Tensor, rois: Tensor
    ) -> List[Tensor]:
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
        self._attrs["inputs"] = [p2, p3, p4, p5, rois]
        x = p2
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output
