#!/usr/bin/env python3
import datetime
import glob
import io
import os

import pandas as pd

# import numpy as np
import sqlalchemy
from sshtunnel import SSHTunnelForwarder


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def get_logfiles():
    path = r"./**/*.log"
    files = glob.glob(path, recursive=True)
    files.sort()
    return files


def get_log_params(logfile):
    branch_name = " "
    commit = " "
    node_id = " "
    gpu_arch = " "
    compute_units = 0
    ngpus = 0
    rocm_vers = " "
    compiler_vers = "release"
    for line in open(logfile):
        if "git_branch" in line:
            lst = line.split()
            branch_name = lst[1]
        if "commit" in line:
            lst = line.split()
            commit = lst[1]
        if "hostname" in line:
            lst = line.split()
            node_id = lst[1]
        if "GPU_arch" in line:
            lst = line.split()
            gpu_arch = lst[2]
        if "Name:                    gfx" in line:
            ngpus = ngpus + 1
        if "Compute Unit" in line:
            lst = line.split()
            compute_units = lst[2]
        if "InstalledDir" in line:
            lst = line.split()
            rocm_vers = lst[1][
                lst[1].find("/opt/rocm-")
                + len("/opt/rocm-") : lst[1].rfind("/llvm/bin")
            ]
    return (
        branch_name,
        commit,
        node_id,
        gpu_arch,
        compute_units,
        ngpus,
        rocm_vers,
        compiler_vers,
    )


def parse_logfile(files):
    # glue = ""
    res = []
    # tests = []
    for logfile in files:
        if "resnet50" in logfile or "vit.log" in logfile:
            init_bs = 0
            for line in open(logfile):
                if "batch_size:" in line:
                    lst = line.split()
                    lst[1] = int(lst[1].replace(",", ""))
                    if (
                        lst[1] > init_bs
                    ):  # only grab first 9 results for different batch sizes from these tests
                        init_bs = lst[1]
                        res.append(lst[3])
        if "bert.log" in logfile:
            for line in open(logfile):
                if "batch_size:" in line:  # grab all 45 results from these tests
                    lst = line.split()
                    res.append(lst[5])
        if "sdiff.log" in logfile:
            for line in open(logfile):
                if "sd e2e:" in line:  # results for stable diffusion
                    lst = line.split()
                    res.append(lst[2])
    return res


def get_baseline(table, connection):
    query = (
        """SELECT * from """
        + table
        + """ WHERE Datetime = (SELECT MAX(Datetime) FROM """
        + table
        + """ where git_branch='amd-develop' );"""
    )
    return pd.read_sql_query(query, connection)


def store_new_test_result(
    table_name,
    test_results,
    testlist,
    node_id,
    branch_name,
    commit,
    gpu_arch,
    compute_units,
    ngpus,
    rocm_vers,
    compiler_vers,
    connection,
):
    params = [
        str(node_id),
        str(branch_name),
        str(commit),
        str(gpu_arch),
        compute_units,
        ngpus,
        str(rocm_vers),
        str(compiler_vers),
        str(datetime.datetime.now()),
    ]
    df = pd.DataFrame(
        data=[params],
        columns=[
            "hostname",
            "git_branch",
            "git_commit",
            "GPU_arch",
            "Compute_units",
            "number_of_gpus",
            "ROCM_version",
            "compiler_version",
            "Datetime",
        ],
    )
    df_add = pd.DataFrame(data=[test_results], columns=testlist)
    df = pd.concat([df, df_add], axis=1)
    print("new test results dataframe:", df)
    df.to_sql(table_name, connection, if_exists="append", index=False)
    return 0


def compare_test_to_baseline(baseline, test, testlist):
    regression = 0
    if not baseline.empty:
        base = baseline[testlist].to_numpy(dtype="float")
        base_list = base[0]
        ave_perf = 0
        for i in range(len(base_list)):
            # success criterion:
            if base_list[i] > 1.01 * float(test[i]):
                print(
                    "test # ",
                    i,
                    "shows regression by {:.3f}%".format(
                        (float(test[i]) - base_list[i]) / base_list[i] * 100
                    ),
                )
                regression = 1
            if base_list[i] > 0:
                ave_perf = ave_perf + float(test[i]) / base_list[i]
        if regression == 0:
            print("no regressions found")
        ave_perf = ave_perf / len(base_list)
        print("average performance relative to baseline:", ave_perf)
    else:
        print("could not find a baseline")
    return regression


def main():
    files = get_logfiles()
    results = []
    baseline = []
    testlist = []
    # parse the test parameters from the logfile
    for filename in files:
        (
            branch_name,
            commit,
            node_id,
            gpu_arch,
            compute_units,
            ngpus,
            rocm_vers,
            compiler_vers,
        ) = get_log_params(filename)

    print("Branch name:", branch_name)
    print("Git_commit:", commit)
    print("Node name:", node_id)
    print("GPU_arch:", gpu_arch)
    print("Compute units:", compute_units)
    print("ROCM_version:", rocm_vers)
    print("Compiler_version:", compiler_vers)
    # parse results, get the Tflops value for "Best Perf" kernels
    results = parse_logfile(files)

    print("Number of tests:", len(results))
    sql_hostname = "127.0.0.1"
    sql_username = os.environ["dbuser"]
    sql_password = os.environ["dbpassword"]
    sql_main_database = "sys"
    sql_port = 3306
    hostname = os.uname()[1]
    if hostname == "jwr-amd-132":
        sqlEngine = sqlalchemy.create_engine(
            "mysql+pymysql://{0}:{1}@{2}/{3}".format(
                sql_username, sql_password, sql_hostname, sql_main_database
            )
        )
        conn = sqlEngine.connect()
    else:
        ssh_host = os.environ["dbsship"]
        ssh_user = os.environ["dbsshuser"]
        ssh_port = int(os.environ["dbsshport"])
        ssh_pass = os.environ["dbsshpassword"]
        with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_pass,
            remote_bind_address=(sql_hostname, sql_port),
        ) as tunnel:
            sqlEngine = sqlalchemy.create_engine(
                "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
                    sql_username,
                    sql_password,
                    sql_hostname,
                    tunnel.local_bind_port,
                    sql_main_database,
                )
            )
        conn = sqlEngine.connect()
    # save gemm performance tests:
    for i in range(1, len(results) + 1):
        testlist.append("Test%i" % i)
    table_name = "ait_performance"

    baseline = get_baseline(table_name, conn)
    store_new_test_result(
        table_name,
        results,
        testlist,
        node_id,
        branch_name,
        commit,
        gpu_arch,
        compute_units,
        ngpus,
        rocm_vers,
        compiler_vers,
        conn,
    )
    conn.close()

    # compare the results to the baseline if baseline exists
    regression = 0
    regression = compare_test_to_baseline(baseline, results, testlist)
    return regression


if __name__ == "__main__":
    main()
