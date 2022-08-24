# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import datetime
import json
import os
import re
from typing import Any, List

import click
import jinja2
import xmltodict
from aitemplate.utils.markdown_table import markdownTable


REPORT_TITLE_TEMPLATE = jinja2.Template("""[{{date}}] CUDA Report""")
REPORT_BODY_TEMPLATE = jinja2.Template(
    """
# Environment

{{git_commit}}

# Unittest Results

{{unittest}}

# Performance Benchmarks

{{benchmark}}

"""
)


def num2str(num: float) -> str:
    return "%.3f" % num


def gen_table(data: List[Any]) -> str:
    if len(data) > 0:
        table = markdownTable(data).getMarkdown().replace("```", "")
    else:
        table = ""
    out = f"\n```{table}\n```\n"
    return out


def submit_report(title, body, token=None):
    from github import Github

    if token is None:
        token = os.environ.get("GITHUB_TOKEN", None)
    if token is None:
        raise RuntimeError("Can't find Github Token")
    g = Github(token)
    repo = g.get_repo("fairinternal/AITemplate")
    _ = repo.create_issue(
        title=title,
        body=body,
        assignee="ipiszy",
        labels=[repo.get_label("Nightly Report")],
    )


def get_git_commit(ait_path: str) -> str:
    if os.environ.get("NIGHTLY", None) is not None:
        with open("/AITemplate/COMMIT_INFO") as fi:
            commit_id = fi.readline().replace("\n", "")
            commit_author = fi.readline().replace("\n", "")
            commit_time = fi.readline().replace("\n", "")
    else:
        import git

        repo = git.Repo(ait_path)
        commit_id = repo.head.object.hexsha
        commit_author = repo.head.commit.author
        commit_time = repo.head.commit.authored_datetime

    output = f"\nCommit: {commit_id}\nAuthor: {commit_author}\nCommit Time: {commit_time}\n\n"  # noqa: E501
    return output


def parse_unittest_xml(log_path) -> str:
    xml_data = ""
    output = ""
    failed_test = []
    skipped_test = []
    if os.environ.get("NIGHTLY", None) is not None:
        with open("/AITemplate/unittest_result.xml") as fi:
            xml_data = fi.read()
    else:
        if os.path.exists(log_path) is False:
            return "\nCouldn't not find pytest XML reports.\n"
        LOG_FLAG = re.compile(r"XML_REPORT_START")
        SEP_FLAG = re.compile(r"=+")
        with open(log_path) as fi:
            lines = fi.readlines()
            enter_log = False
            enter_json = False
            for line in lines:
                if not enter_log:
                    if LOG_FLAG.match(line) is not None:
                        enter_log = True
                else:
                    if not enter_json:
                        if SEP_FLAG.match(line) is not None:
                            enter_json = True
                    else:
                        if SEP_FLAG.match(line) is None:
                            xml_data += line
                        else:
                            enter_json = False
                            enter_log = False  # lazy
    if len(xml_data) == 0:
        return "\nCouldn't not find pytest XML reports.\n"
    data = xmltodict.parse(xml_data)

    items = data["testsuites"]["testsuite"]["testcase"]
    for entry in items:
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

    output += "\n## Failed Tests\n"
    output += gen_table(failed_test)

    output += "\n## Skipped Tests\n"
    output += gen_table(skipped_test)

    return output


def parse_unittest_markdown(log_path) -> str:
    output = ""
    if os.environ.get("NIGHTLY", None) is not None:
        with open("/AITemplate/unittest_report.md") as fi:
            output = fi.read()
    else:
        if os.path.exists(log_path) is False:
            return "\nCouldn't not find pytest reports.\n"
        LOG_FLAG = re.compile(r"MARKDOWN_REPORT_START")
        SEP_FLAG = re.compile(r"=+")
        with open(log_path) as fi:
            lines = fi.readlines()
            enter_log = False
            enter_json = False
            for line in lines:
                if not enter_log:
                    if LOG_FLAG.match(line) is not None:
                        enter_log = True
                else:
                    if not enter_json:
                        if SEP_FLAG.match(line) is not None:
                            enter_json = True
                    else:
                        if SEP_FLAG.match(line) is None:
                            if line[0] == "\n":
                                line = line[1:]
                            output += line
                        else:
                            enter_json = False
                            enter_log = False  # lazy
    if len(output) == 0:
        return "\nCouldn't not find pytest markdown reports.\n"
    return output


def parse_benchmark_json(log_path: str) -> str:
    json_data = ""
    output = ""
    ads_data = []
    cv_data = []
    resnet_data = []
    if os.environ.get("NIGHTLY", None) is not None:
        with open("/AITemplate/benchmark_result.json") as fi:
            json_data = fi.read()
    else:
        if os.path.exists(log_path) is False:
            return "\nCouldn't not find pytest XML reports.\n"
        LOG_FLAG = re.compile(r"JSON_REPORT_START")
        SEP_FLAG = re.compile(r"=+")
        with open(log_path) as fi:
            lines = fi.readlines()
            enter_log = False
            enter_json = False
            for line in lines:
                if not enter_log:
                    if LOG_FLAG.match(line) is not None:
                        enter_log = True
                else:
                    if not enter_json:
                        if SEP_FLAG.match(line) is not None:
                            enter_json = True
                    else:
                        if SEP_FLAG.match(line) is None:
                            json_data += line
                        else:
                            enter_json = False
                            enter_log = False  # lazy
    if len(json_data) == 0:
        return "\nCouldn't not find benchmark results.\n"
    data = json.loads(json_data)
    for entry in data:
        if entry["report_template"] == "inline-cvr":  # Ads model
            ads_data.append(
                {
                    "Model": entry["task"],
                    "Batch Size": entry["batch_size"],
                    "FP16 ACC": entry["fp16_acc"],
                    "Use Group Gemm": entry["use_group_gemm"],
                    "Graph Mode": entry["graph_mode"],
                    "Latency": num2str(entry["latency"]),
                    "QPS": num2str(entry["qps"]),
                    "Num Threads": num2str(entry["num_threads"]),
                }
            )
        elif entry["report_template"] == "resnet":  # Resnet model
            resnet_data.append(
                {
                    "Model": entry["task"],
                    "Batch Size": entry["batch_size"],
                    "FP16 ACC": entry["fp16_acc"],
                    "Graph Mode": entry["graph_mode"],
                    "Latency": num2str(entry["latency"]),
                    "QPS": num2str(entry["qps"]),
                }
            )
        elif entry["report_template"] == "detection":  # CV model
            cv_data.append(
                {
                    "Model": entry["task"],
                    "Batch Size": entry["batch_size"],
                    "Graph Mode": entry["graph_mode"],
                    "Input Size": entry["input_size"],
                    "Box ACC": entry["boxes_acc"],
                    "Score ACC": entry["scores_acc"],
                    "Class ACC": entry["classes_acc"],
                    "Latency": num2str(entry["latency"]),
                    "FPS": num2str(entry["fps"]),
                }
            )
    output += "\n## Ads Models\n"
    output += gen_table(ads_data)

    output += "\n## CV Models\n"
    output += gen_table(cv_data)

    output += "\n## ResNet Models\n"
    output += gen_table(resnet_data)

    return output


@click.command()
@click.option("--code-path", default="", help="Git repo of AITemplate")
@click.option(
    "--unittest-log",
    default="",
    help="LOG file path contains pytest XML reports",  # noqa: E501
)
@click.option(
    "--benchmark-log",
    default="",
    help="LOG file path contains benchmark JSON reports",  # noqa: E501
)
@click.option(
    "--github-token", default="", help="Github Token to publish issue"
)  # noqa: E501
def generate_report(
    code_path: str = "",
    unittest_log: str = "",
    benchmark_log: str = "",
    github_token: str = "",
) -> str:
    date = datetime.datetime.now().date()
    git_commit = get_git_commit(code_path)
    unittest = parse_unittest_xml(unittest_log)
    benchmark = parse_benchmark_json(benchmark_log)

    title = REPORT_TITLE_TEMPLATE.render(date=date)
    body = REPORT_BODY_TEMPLATE.render(
        git_commit=git_commit, unittest=unittest, benchmark=benchmark
    )
    # print(title)
    # print(body)
    submit_report(title, body, github_token)


if __name__ == "__main__":
    generate_report()
