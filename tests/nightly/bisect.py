# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Bisect tools to find commit with bug
"""
import os
from subprocess import check_output
from typing import List

from aitemplate.utils import logger


def green_text(text):
    return f"\033[92m{text}\033[0m"


def cyan_text(text):
    return f"\033[96m{text}\033[0m"


def warning_text(text):
    return f"\033[93m{text}\033[0m"


def extract_candidate(
    old_commit_id: str, new_commit_id: str = None, max_commits: int = 101
) -> List[str]:
    commits = []
    if new_commit_id is None:
        new_commit_id = check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")
    new_commit_id = new_commit_id.replace("\n", "")
    commits.append(new_commit_id)
    found_old_commit = False
    for i in range(1, max_commits):
        commit_id = check_output(["git", "rev-parse", f"HEAD~{i}"]).decode("utf-8")
        commit_id = commit_id.replace("\n", "")
        commits.append(commit_id)
        if commit_id == old_commit_id:
            found_old_commit = True
            break
    if found_old_commit:
        logger.info("Bisect", "Found Old Commits. Total Commits: %d" % len(commits))
    else:
        logger.info(
            "Bisect", "Couldn't find old Commits. Total Commits: %d" % len(commits)
        )
    return commits


def switch_commit(commit_id: str) -> None:
    check_output(["git", "checkout", commit_id])
    logger.info("Bisect", f"\033[91mSwitch to {commit_id}\033[0m")


def main() -> None:
    old_commit = input("Old Commit ID: ")
    new_commit = input("New Commit ID: ")
    exec_command = input("Test Command: ")
    commits = extract_candidate(old_commit, new_commit)
    check_output(["git", "reset", "--hard", new_commit])
    lo = 0
    hi = len(commits) - 1
    stop_flag = False
    while lo <= hi and not stop_flag:
        mid = (lo + hi) // 2
        current_commit = commits[mid]
        switch_commit(current_commit)
        os.system(exec_command)
        print(warning_text("=" * 60))
        output = check_output(
            ["git", "show", "-s", '--format="[%ci]: %s"', current_commit]
        ).decode("utf-8")
        logger.info("Bisect", f"Current commit: {output}")
        action = ""
        while action != "n" and action != "o" and action != "s":
            action = input(
                green_text("Move to [o]lder commit or [n]ewer commit, or [s]top: ")
            )
        if action == "n":
            hi = mid - 1
        elif action == "o":
            lo = mid + 1
        else:
            stop_flag = True
        print(warning_text("=" * 60))


if __name__ == "__main__":
    main()
