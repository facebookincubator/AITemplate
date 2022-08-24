# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] default logger
"""
import logging


def info(name, message):
    """[summary]

    Parameters
    ----------
    message : [type]
        [description]
    """
    logger = logging.getLogger(name)
    logger.info(message)


def debug(name, message):
    """[summary]

    Parameters
    ----------
    message : [type]
        [description]
    """
    logger = logging.getLogger(name)
    logger.debug(message)


def warning(name, message):
    """[summary]

    Parameters
    ----------
    message : [type]
        [description]
    """
    logger = logging.getLogger(name)
    logger.warning(message)


def is_debug():
    logger = logging.getLogger("aitemplate")
    return logger.level == logging.DEBUG
