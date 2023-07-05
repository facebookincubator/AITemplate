import logging
import os
import torch
import importlib.machinery

logger: logging.Logger = logging.getLogger(__name__)


def is_oss_ait_model():
    return False


def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin


try:
    torch.ops.load_library("//deeplearning/ait:AITModel")
    logger.info("===Load non-OSS AITModel===")

except (ImportError, OSError):
    lib_path = _get_extension_path("libait_model")
    torch.ops.load_library(lib_path)
    logger.info("===Load OSS AITModel===")

    def is_oss_ait_model():  # noqa: F811
        return True
