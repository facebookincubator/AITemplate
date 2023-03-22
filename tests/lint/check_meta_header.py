#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Check Python source code contains Meta copyright header
"""

from __future__ import annotations

import os
import sys
from typing import List

import click


def process_header(header, comment):
    lines = header.split("\n")
    new_lines = [comment + " " + line for line in lines]
    return "\n".join(new_lines) + "\n"


with open("licenses/license.header.txt") as fi:
    HEADER = fi.read()
HEADER_lines = HEADER.splitlines()[1:]
PY_HEADER = process_header(HEADER, "#")
CPP_HEADER = process_header(HEADER, "//")


def dfs(root_path: str) -> List[str]:
    """DFS source code tree to find python files missing header

    Parameters
    ----------
    root_path : str
        root source directory path

    Returns
    -------
    List[str]
        file list missing header
    """
    ret = []
    for root, _, files in os.walk(root_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if path.endswith(".py"):
                with open(path) as fi:
                    src = fi.read()
                    flag = True
                    for line in HEADER_lines:
                        if line not in src:
                            flag = False
                            break
                    if not flag:
                        ret.append(path)
    return ret


def fix_header(file_list: List[str]) -> None:
    """Adding Meta header to to source files

    Parameters
    ----------
    file_list : List[str]
        file list missing header
    """
    for path in file_list:
        src = ""
        with open(path) as fi:
            src = fi.read()
        with open(path, "w") as fo:
            fo.write(PY_HEADER)
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
