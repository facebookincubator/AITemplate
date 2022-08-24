# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] roi_ops op
"""
from .roi_ops import roi_ops_base


# pylint: disable=C0103
class roi_align(roi_ops_base):
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
        self._attrs["op"] = "roi_align"
