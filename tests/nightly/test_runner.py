# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
A subprocess based multiple GPUs runner for unit tests
"""

from __future__ import annotations

import os

import typing

import click
import xmltodict
from aitemplate.backend.target import Target
from aitemplate.backend.task_runner import BaseRunner, Task
from aitemplate.testing import detect_target

from aitemplate.utils import logger
from aitemplate.utils.markdown_table import markdownTable


def gen_table(data: typing.List[typing.Any]) -> str:
    out = "\n```"
    if len(data) > 0:
        out += markdownTable(data).getMarkdown().replace("```", "")
    out += "\n```\n"
    return out


def process_task(task: Task) -> None:
    """Extract test results from XML reports

    Parameters
    ----------
    task : Task
        A unit test task
    """
    logger.info(__name__, "Process: {file}".format(file=task._idx))
    ret = {}
    if os.path.exists(task._idx):
        with open(task._idx) as fi:
            xml_data = fi.read()
        data = xmltodict.parse(xml_data)
        failed_test = []
        skipped_test = []
        try:
            items = data["testsuites"]["testsuite"]["testcase"]
        except KeyError:
            items = {
                "@classname": os.path.basename(task._idx[:-3] + ".py"),
                "skipped": True,
                "@name": "",
            }

        def proc_single_case(entry):
            if "failure" in entry:
                failed_test.append(
                    {
                        "Test Class": entry["@classname"],
                        "Test Name": entry["@name"],
                    }
                )
            if "skipped" in entry:
                skipped_test.append(
                    {
                        "Test Class": entry["@classname"],
                        "Test Name": entry["@name"],
                    }
                )

        if isinstance(items, dict):
            proc_single_case(items)
        elif isinstance(items, list):
            for item in items:
                proc_single_case(item)
        else:
            raise RuntimeError("Unknown output XML format.")
        ret["failed_test"] = failed_test
        ret["skipped_test"] = skipped_test
        task._ret = ret
    else:
        task._failed = True


def process_return(task: Task):
    """Generate profile result from a profiling task

    Parameters
    ----------
    task : Task
        A profiling task

    Returns
    -------
    Tuple[Union[int, str], ProfileResult]
        Tuple of task idx (usually the algorithm name/id) and profiling result
    """
    return (task._idx, task._ret)


class Runner(BaseRunner):
    def __init__(self, devs: list[int], tag: str = "runner", timeout: int = 1800):
        super().__init__(devs, tag, timeout)
        logger.info(__name__, "Using {n} GPU for unit testing".format(n=len(devs)))
        self._dev_flag = Target.current().dev_select_flag()
        self._ftask_proc = process_task
        self._fret_proc = process_return

    def push(self, idx: str, cmd: str):
        self._queue.append(
            Task(idx, cmd, self._tag, dev_flag=self._dev_flag, shell=True)
        )

    def pull(self):
        ret = super().pull(self._ftask_proc, self._fret_proc)
        return ret


def gen_tasks(unittest_path: str) -> typing.Tuple[str, str]:
    tasks = []
    for root, _, files in os.walk(unittest_path, topdown=False):
        for file in files:
            if file.endswith(".py"):
                test_path = str(os.path.join(root, file))
                report_path = test_path[:-2] + "xml"
                tasks.append((test_path, report_path))

    return tasks


@click.command()
@click.option(
    "--unittest-path",
    default="/AITemplate/tests/unittest",
    help="AITemplate unittest folder",
)
@click.option(
    "--devs", default="0,1,2,3,4,5,6,7", help="GPU ids will be used for tests"
)
@click.option(
    "--output-path", default="/AITemplate/unittest_report.md", help="Output report path"
)
def main(unittest_path: str, devs: str, output_path: str):
    devs = [int(x) for x in devs.split(",")]
    target = detect_target()

    failed_test = []
    skipped_test = []

    with target:
        runner = Runner(devs=devs, timeout=1800)
        tasks = gen_tasks(unittest_path)
        logger.info(__name__, "Running {n} tests".format(n=len(tasks)))

        for test_path, report_path in tasks:
            cmd = f'pytest {test_path} --junitxml="{report_path}"'
            runner.push(report_path, cmd)
        runner.join()
        results = runner.pull()
    for _, ret in results:
        failed_test.extend(ret["failed_test"])
        skipped_test.extend(ret["skipped_test"])

    output = ""
    output += "\n## Failed Tests\n"
    output += gen_table(failed_test)

    output += "\n## Skipped Tests\n"
    output += gen_table(skipped_test)

    with open(output_path, "w") as fo:
        fo.write(output)


if __name__ == "__main__":
    main()
