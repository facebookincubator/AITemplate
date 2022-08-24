# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Check Python source code contains Meta copyright header 
"""

from __future__ import annotations

import os
import sys

import click

HEADER = "# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.\n"


def dfs(root_path: str) -> list[str]:
    """DFS source code tree to find python files missing header

    Parameters
    ----------
    root_path : str
        root source directory path

    Returns
    -------
    list[str]
        file list missing header
    """
    ret = []
    for root, _, files in os.walk(root_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if path.endswith(".py"):
                with open(path) as fi:
                    line = fi.readline()
                    if line != HEADER:
                        ret.append(path)
    return ret


def fix_header(file_list: list[str]) -> None:
    """Adding Meta header to to source files

    Parameters
    ----------
    file_list : list[str]
        file list missing header
    """
    for path in file_list:
        src = ""
        with open(path) as fi:
            src = fi.read()
        with open(path, "w") as fo:
            fo.write(HEADER)
            fo.write(src)


@click.command()
@click.option(
    "--path", help="Root directory of source to be checked", required=True, type=str
)
@click.option(
    "--fixit", default=False, help="Fix missing header", required=False, type=bool
)
def check_header(path, fixit):
    ret = dfs(path)
    if len(ret) == 0:
        sys.exit(0)
    print("Need to add Meta header to the following files.")
    print("----------------File List----------------")
    for line in ret:
        print(line)
    print("-----------------------------------------")
    if fixit:
        fix_header(ret)
    sys.exit(1)


if __name__ == "__main__":
    check_header()
