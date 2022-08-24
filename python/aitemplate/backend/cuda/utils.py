# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from aitemplate.utils.mk_cutlass_lib.mk_cutlass_lib import mk_cutlass_lib

from ...utils import logger
from .. import registry

# pylint: disable=C0103,C0415,W0707


class Args(object):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self, arch):
        """[summary]

        Parameters
        ----------
        arch : [type]
            [description]
        """
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.generator_target = ""
        self.architectures = arch
        self.kernels = "all"
        self.ignore_kernels = ""
        self.cuda_version = "11.4.0"
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True


registry.reg("cuda.make_cutlass_lib")(mk_cutlass_lib)


@registry.reg("cuda.gen_cutlass_ops")
def gen_ops(arch):
    """[summary]

    Parameters
    ----------
    arch : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    import cutlass_lib

    args = Args(arch)
    manifest = cutlass_lib.manifest.Manifest(args)
    try:
        func = getattr(cutlass_lib.generator, "GenerateSM" + arch)
        func(manifest, args.cuda_version)
    except AttributeError:
        raise NotImplementedError(
            "Arch " + arch + " is not supported by current cutlass lib."
        )
    try:
        func = getattr(cutlass_lib.extra_operation, "GenerateSM" + arch)
        func(manifest, args)
    except AttributeError:
        logger.warning(__file__, "Arch " + arch + " is not supported by extra ops.")
    return manifest.operations
