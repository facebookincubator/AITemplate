# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...compiler.ops import multi_level_roi_align, roi_align
from .module import Module


class RoiAlign(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
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
    ):
        """[summary]

        Parameters
        ----------
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        padding : [type]
            [description]
        """
        super().__init__()
        self.op = roi_align(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
        )

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) == 2
        x = args[0]
        rois = args[1]
        return self.op(x, rois)


class FPNRoiAlign(Module):
    """[summary]

    Parameters
    ----------
    Module : [type]
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
    ):
        """[summary]

        Parameters
        ----------
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        padding : [type]
            [description]
        """
        super().__init__()
        self.op = multi_level_roi_align(
            num_rois,
            pooled_size,
            sampling_ratio,
            spatial_scale,
            position_sensitive,
            continuous_coordinate,
            im_shape,
        )

    def forward(self, *args):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        assert len(args) >= 2
        x = args[0]
        rois = args[1]
        return self.op(x, rois)
