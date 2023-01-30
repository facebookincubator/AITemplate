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
"""
A utility script used for modify entries in the profiling cache
"""

import argparse
import hashlib
import logging
import os
import re
import sqlite3

from typing import Any, Dict, List, Tuple

import jinja2
from aitemplate.backend.profiler_cache import GEMM_INSERT_TEMPLATE, GEMM_QUERY_TEMPLATE


logging.basicConfig(format="%(name)s: %(message)s", level=logging.INFO)

_LOGGER = logging.getLogger("update-cache")

DEFAULT_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT id, op_type, algo, device, exec_entry, dtype_acc
FROM {{table}}
WHERE
{% if has_device %}
device='{{device}}' AND
{% endif %}
{% if has_exec_entry_sha1 %}
exec_entry_sha1='{{exec_entry_sha1}}' AND
{% endif %}
{% if has_algo %}
algo='{{algo}}' AND
{% endif %}
{% if has_dtype_acc %}
dtype_acc='{{dtype_acc}}' AND
{% endif %}
op_type='{{op_type}}'
"""
)

ID_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT id, op_type, algo, device, exec_entry, dtype_acc
FROM {{table}}
WHERE
id={{id}}
"""
)

DEVICE_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT id, op_type, algo, device, exec_entry, dtype_acc
FROM {{table}}
WHERE
device='{{device}}'
"""
)

DEVICE_ONE_OP_TYPE_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT id, op_type, algo, device, exec_entry, dtype_acc
FROM {{table}}
WHERE
op_type='{{op_type}}' AND
device='{{device}}'
"""
)

QUERY_ROW_TEMPLATE = jinja2.Template(
    """
SELECT *
FROM {{table}}
WHERE
id={{id}}
"""
)


QUERY_COLUMN_NAMES_TEMPLATE = jinja2.Template(
    """
SELECT * FROM {{table}}
"""
)


DEL_ID_TEMPLATE = jinja2.Template(
    """
DELETE FROM {{table}} WHERE id={{id}}
"""
)


def del_entry(
    db_conn: sqlite3.Connection,
    table: str,
    entry_id: str,
    op_type: str,
):
    """create an sm75 entry based on 'entry' and insert the newly-created
       entry into the db based

    Parameters
    ----------
    db_conn: sqlite3.Connection
        sqlite3 Connection object
    table:
        db table
    entry_id:
        entry id to be deleted
    op_type: str
        op type in the query, e.g. gemm_rcr_bias

    Returns
    ----------
    """
    _LOGGER.info("query entry for deletion - id: %s, op_type: %s", entry_id, op_type)
    db_conn_cur = db_conn.cursor()
    entries = query_cache(
        db_conn_cur=db_conn_cur,
        table=table,
        op_type=op_type,
        op_keys=None,
        query_template=ID_QUERY_TEMPLATE,
        algo=None,
        device=None,
        entry_id=entry_id,
    )
    if len(entries) == 0:
        _LOGGER.info("Could not find valid entries, skip")
        return

    assert len(entries) == 1
    if op_type != entries[0][1]:
        raise RuntimeError(
            f"cannot delete the entry, unmatched op_type: {op_type}, {entries[0][1]}"
        )
    _LOGGER.info("deleting entry - id: %s, op_type: %s", entry_id, op_type)
    del_query = DEL_ID_TEMPLATE.render(
        table=table,
        id=entry_id,
    )
    db_conn_cur.execute(del_query)
    db_conn.commit()
    _LOGGER.info("entry deleted successfully")


def insert_sm75_entry(
    db_conn: sqlite3.Connection,
    table: str,
    entry: Tuple[Any],
    column_names: List[str],
    algo: str,
):
    """create an sm75 entry based on 'entry' and insert the newly-created
       entry into the db based

    Parameters
    ----------
    db_conn: sqlite3.Connection
        sqlite3 Connection object
    table:
        db table
    entry: Tuple[Any]
        sm80 entry
    column_names: List[str
        all column names in the table
    algo: str
        new algo used to update the args Dict
    device: str
        new device used to update the args Dict

    Returns
    ----------
    """
    db_conn_cur = db_conn.cursor()
    row_query = QUERY_ROW_TEMPLATE.render(table=table, id=entry[0])
    db_conn_cur.execute(row_query)
    row_values = db_conn_cur.fetchall()
    assert len(row_values) == 1
    args = make_insertion_args(column_names, row_values[0])

    if algo is None:
        algo = make_75_algo_from_80(args["algo"])

    args["algo"] = algo
    args["device"] = "75"

    new_args_str = "\n".join(["{}: {}".format(n, v) for n, v in args.items()])
    _LOGGER.info("new_args:\n%s", new_args_str)
    query_sql = GEMM_QUERY_TEMPLATE.render(
        dev="cuda",
        dtype_a=args["dtype_a"],
        dtype_b=args["dtype_b"],
        dtype_c=args["dtype_c"],
        dtype_acc=args["dtype_acc"],
        major_a=args["major_a"],
        major_b=args["major_b"],
        major_c=args["major_c"],
        op_type=args["op_type"],
        device=args["device"],
        epilogue=args["epilogue"],
        split_k=args["split_k"],
        exec_entry_sha1=args["exec_entry_sha1"],
    )

    db_conn_cur.execute(query_sql)
    existing_entries = db_conn_cur.fetchall()
    # make sure we don't overwrite existing entries
    if len(existing_entries) != 0:
        id_query = DEFAULT_QUERY_TEMPLATE.render(
            table=table,
            op_type=args["op_type"],
            device=args["device"],
            exec_entry_sha1=args["exec_entry_sha1"],
        )
        db_conn_cur.execute(id_query)
        id_entries = db_conn_cur.fetchall()
        raise RuntimeError(
            "An entry (id: {}, op_type: {}) already exist in the table: {}".format(
                id_entries[0][0], args["op_type"], existing_entries[0][0]
            )
        )

    insertion_sql = GEMM_INSERT_TEMPLATE.render(dev="cuda", **args)
    db_conn_cur.execute(insertion_sql)
    db_conn.commit()
    _LOGGER.info(
        "successfully insert an sm75 entry for: '%s', '%s'",
        args["op_type"],
        args["exec_entry"],
    )


def make_gemm_exec_key(op_keys: str) -> str:
    """make exec_key for gemm-family ops based on the given op_keys string

    Parameters
    ----------
    op_keys: str
        op_keys used to make the exec_key, e.g. 2x3x4

    Returns
    ----------
    str
        exec_key
    """
    if ";" in op_keys:
        raise RuntimeError("invalid op_keys for gemm: '{}'".format(op_keys))

    values = [int(v) for v in op_keys.split("x")]
    if len(values) != 3:
        raise RuntimeError("invalid op_keys for gemm: '{}'".format(op_keys))

    name_values = [("M", values[0]), ("N", values[1]), ("K", values[2])]
    return " && ".join(["{} == {}".format(n, v) for n, v in name_values])


def make_bmm_exec_key(op_keys: str) -> str:
    """make exec_key for bmm-family ops based on the given op_keys string

    Parameters
    ----------
    op_keys: str
        op_keys used to make the exec_key, e.g. 2x3x4x5

    Returns
    ----------
    str
        exec_key
    """
    if ";" in op_keys:
        raise RuntimeError("invalid op_keys for bmm: '{}'".format(op_keys))

    values = [int(v) for v in op_keys.split("x")]
    if len(values) != 4:
        raise RuntimeError("invalid op_keys for bmm: '{}'".format(op_keys))

    name_values = [
        ("B", values[0]),
        ("M", values[1]),
        ("N", values[2]),
        ("K", values[3]),
    ]
    return " && ".join(["{} == {}".format(n, v) for n, v in name_values])


def make_group_gemm_exec_key(op_keys: str) -> str:
    """make exec_key for group_gemm-family ops based on the given op_keys string

    Parameters
    ----------
    op_keys: str
        op_keys used to make the exec_key, e.g. 2x3x4;4x5x6;7x8x9

    Returns
    ----------
    str
        exec_key
    """
    shapes = op_keys.split(";")
    name_value_strs = []
    for i, shape in enumerate(shapes):
        values = shape.split("x")
        if len(values) != 3:
            raise RuntimeError("invalid op_keys for group_gemm: '{}'".format(op_keys))
        name_values = [
            ("GROUP_{}_M".format(i), int(values[0])),
            ("GROUP_{}_N".format(i), int(values[1])),
            ("GROUP_{}_K".format(i), int(values[2])),
        ]
        name_value_str = " &&".join(["{} == {}".format(n, v) for n, v in name_values])
        name_value_strs.append(" " + name_value_str)
    return "  &&".join(name_value_strs)


def query_cache(
    db_conn_cur: sqlite3.Cursor,
    table: str,
    op_type: str,
    op_keys: str,
    query_template: str,
    algo: str = None,
    device: str = None,
    entry_id: int = None,
    suppress_print: bool = False,
    exec_key: str = None,
    dtype_acc: str = None,
) -> List[Tuple[int, str, str, str, str, str]]:
    """query the database for the given op_type and op_keys in the table and
       print out the entries if the query succeeds.

    Parameters
    ----------
    db_conn_cur: sqlite3.Cursor
        sqlite3 connection Cursor object
    table: str
        the table to be queried
    op_type: str
        op type in the query, e.g. gemm_rcr_bias
    op_keys: str
        op keys used to construct exec_hash_sh1 in the query, e.g. 2x3x4 for
        gemm-family op type
    query_template: str
        jinja template to be rendered to create the query
    device: str optional
        specify the device field in the query if it's not None. Default value
        is None, which means we will query all device kinds.
    algo: str optional
        specify the also field in the query if it's not None. Default value
        is None.
    entry_id: int optional
        specify the id field in the query if it's not None
    suppress_print: bool optional
        do not print returned queries if it's True
    exec_key: str optional
        we will use this exec_key for query if it's not None
    dtype_acc: str optional
        we will use this dtype_acc for query if it's not None

    Returns
    -------
    List[Tuple[int, str, str, str, str, str]]
        returns a list of valid entries
    """
    exec_entry_sha1 = None
    if op_keys is not None and exec_key is None:
        if op_type.startswith("gemm"):
            exec_key = make_gemm_exec_key(op_keys)
        elif op_type.startswith("bmm"):
            exec_key = make_bmm_exec_key(op_keys)
        elif op_type.startswith("group_gemm"):
            exec_key = make_group_gemm_exec_key(op_keys)
        else:
            raise RuntimeError("invalid op_type: " + op_type)

    if exec_key is not None:
        if not suppress_print:
            _LOGGER.info("exec_key: '%s'", exec_key)
        exec_entry_sha1 = hashlib.sha1(exec_key.encode("utf-8")).hexdigest()
        if not suppress_print:
            _LOGGER.info("exec_sha1: '%s'", exec_entry_sha1)

    query_args = {
        "table": table,
        "op_type": op_type,
    }
    query_args["has_exec_entry_sha1"] = False
    if exec_entry_sha1 is not None:
        query_args["exec_entry_sha1"] = exec_entry_sha1
        query_args["has_exec_entry_sha1"] = True

    query_args["has_device"] = False
    if device is not None:
        query_args["device"] = device
        query_args["has_device"] = True

    query_args["has_entry_id"] = False
    if entry_id is not None:
        query_args["id"] = entry_id
        query_args["has_entry_id"] = True

    query_args["has_dtype_acc"] = False
    if dtype_acc is not None:
        query_args["dtype_acc"] = dtype_acc
        query_args["has_dtype_acc"] = True

    query_args["has_algo"] = False
    if algo is not None:
        query_args["algo"] = algo
        query_args["has_algo"] = True

    query = query_template.render(**query_args)
    if not suppress_print:
        print("query:{}".format(query))

    db_conn_cur.execute(query)
    entries = db_conn_cur.fetchall()
    if not suppress_print:
        _LOGGER.info("entries: id, op_type, algo, device, exec_entry")
    for entry in entries:
        if not suppress_print:
            print("entry: {}".format(entry))
    return entries


def process_missing_75_entries_from_80(
    db_conn: sqlite3.Connection,
    table: str,
    op_type: str,
    gen_sm75_entry: bool = False,
):
    """for the given op_type, print out all missing sm75 entries if their
    sm80 counterparts exist in the table. If gen_sm75_entry is True, we will
    generate an sm75 entry based on each returned sm80 entry.

    Parameters
    ----------
    db_conn: sqlite3.Connection
        sqlite3 Connection object
    table:
        db table
    op_type: str
        op type in the query, e.g. gemm_rcr_bias

    Returns
    ----------
    """
    _LOGGER.info("query all missing sm75 entries - op_type: %s", op_type)
    db_conn_cur = db_conn.cursor()
    if op_type == "all":
        op_type = None
        query_template = DEVICE_QUERY_TEMPLATE
    else:
        query_template = DEVICE_ONE_OP_TYPE_QUERY_TEMPLATE
    sm80_entries = query_cache(
        db_conn_cur=db_conn_cur,
        table=table,
        op_type=op_type,
        op_keys=None,
        query_template=query_template,
        algo=None,
        device="80",
        entry_id=None,
        suppress_print=True,
    )

    # entry: (id, op_type, algo, device, exec_entry, dtype_acc)
    for sm80_entry in sm80_entries:
        eid, op_type, _, _, exec_entry, dtype_acc = sm80_entry
        # these ops do not work with sm75
        if op_type.startswith("bmm_softmax") or op_type.startswith("group_gemm"):
            continue
        sm75_entries = query_cache(
            db_conn_cur=db_conn_cur,
            table=table,
            op_type=op_type,
            op_keys=None,
            query_template=DEFAULT_QUERY_TEMPLATE,
            algo=None,
            device="75",
            entry_id=None,
            suppress_print=True,
            exec_key=exec_entry,
            dtype_acc=dtype_acc,
        )
        if len(sm75_entries) == 0:
            print("missing sm75 entry for this sm80 entry: '{}'".format(sm80_entry))
            if gen_sm75_entry:
                _LOGGER.info("gen sm75 entry for: '%s'", sm80_entry)
                column_names = get_column_names(db_conn_cur, table)
                insert_sm75_entry(db_conn, table, sm80_entry, column_names, None)


def make_insertion_args(column_names: List[str], values: List[Any]) -> Dict[str, Any]:
    """generate a dictionary that will be used for inserting an entry.

    Parameters
    ----------
    column_names: List[str]
        column names in the db
    values: List[Any]
        values paired with names

    Return
    ----------
    Dict[str, Any]
        generated dictionary
    """
    if len(column_names) != len(values):
        raise RuntimeError(
            "len(column_names) does not equal to len(values): {}, {}".format(
                len(column_names), len(values)
            )
        )
    name_values = {}
    for name, val in zip(column_names, values):
        name_values[name] = val
    return name_values


def make_75_algo_from_80(old_algo: str):
    """try out best to make an sm75 algo based on the provided sm80 algo.

    Parameters
    ----------
    old_algo: str
        sm80 algo string used for making sm75 algo
    Returns
    -------
    str
        newly-created sm75 algo string
    """
    if "s16816gemm" in old_algo:
        pattern = re.compile(
            r"cutlass_tensorop_(f\d+)_s16816gemm_(f\d+)_(\d+x\d+_\d+)x\d+_([tn]+)_align_\d+_(\d+)"
        )
        result = pattern.match(old_algo)
        if result is None:
            raise RuntimeError("Invalid algo format: '{}'".format(old_algo))

        dtype = result.group(1)
        acc_dtype = result.group(2)
        # make a tile that supported by sm75
        tile = "64x64_32"
        layout = result.group(4)
        align_c = result.group(5)
        new_algo = f"cutlass_tensorop_{dtype}_s1688gemm_{acc_dtype}_{tile}x2_{layout}_align_1_{align_c}"
    elif "s1688gemm" in old_algo:
        pattern = re.compile(
            r"cutlass_tensorop_s1688gemm_(\d+x\d+_\d+)x\d+_([tn]+)_align_\d+_(\d+)"
        )
        result = pattern.match(old_algo)
        if result is None:
            raise RuntimeError("Invalid algo format: '{}'".format(old_algo))

        tile = "64x64_32"
        layout = result.group(2)
        align_c = result.group(3)
        new_algo = f"cutlass_tensorop_h1688gemm_{tile}x2_{layout}_align_1_{align_c}"
    elif "h16816gemm" in old_algo:
        pattern = re.compile(
            r"cutlass_tensorop_h16816gemm_(\d+x\d+_\d+)x\d+_([tn]+)_align_\d+_(\d+)"
        )
        result = pattern.match(old_algo)
        if result is None:
            raise RuntimeError("Invalid algo format: '{}'".format(old_algo))

        tile = "64x64_32"
        layout = result.group(2)
        align_c = result.group(3)
        new_algo = f"cutlass_tensorop_h1688gemm_{tile}x2_{layout}_align_1_{align_c}"
    else:
        raise RuntimeError("Invalid old_algo format: '{}'".format(old_algo))

    _LOGGER.info("new_algo: '%s'", new_algo)
    return new_algo


def get_column_names(db_conn_cur: sqlite3.Cursor, table: str):
    column_names_query = QUERY_COLUMN_NAMES_TEMPLATE.render(table=table)
    columns = db_conn_cur.execute(column_names_query)
    column_names = [col[0] for col in columns.description]
    _LOGGER.info("colum_names:%s", column_names)
    return column_names


def gen_one_75_entry_from_80(
    db_conn: sqlite3.Connection,
    table: str,
    op_type: str,
    op_keys: str,
    algo: str = None,
):
    """generate sm75 entries based on the corresponding sm80 entries queried
    by the given op_type and op_keys.

    Parameters
    ----------
    db_conn: sqlite3.Connection
        sqlite3 Connection object
    table: str
        the table to be queried
    op_type: str
        op type in the query, e.g. gemm_rcr_bias
    op_keys: str
        op keys used to construct exec_hash_sh1 in the query, e.g. 2x3x4 for
        gemm-family op type
    algo: optional str
        if it's not None, the algo field will be filled with the specified algo
        string for the sm75 entry.

    Returns
    -------
    None
    """
    db_conn_cur = db_conn.cursor()
    entries = query_cache(
        db_conn_cur=db_conn_cur,
        table=table,
        op_type=op_type,
        op_keys=op_keys,
        query_template=DEFAULT_QUERY_TEMPLATE,
        algo=None,
        device="80",
    )
    if len(entries) == 0:
        _LOGGER.info("Could not find valid entries, skip")
        return

    column_names = get_column_names(db_conn_cur, table)

    if algo is not None:
        if len(entries) != 1:
            raise RuntimeError(
                "multiple entried found, which cannot be applied to a single algo"
            )
        insert_sm75_entry(db_conn, table, entries[0], column_names, algo)
        return

    for entry in entries:
        insert_sm75_entry(db_conn, table, entry, column_names, algo)


def parse_cmd_arguments():
    """a function to parse command-line arguments

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    arg_parser = argparse.ArgumentParser(
        description="update entries in the profiling database",
        usage="""
Examples:
  * insert an sm75 entry for the given op_type and op_keys from an existing sm80 entry:
    --gen_one_75_entry_from_80 --db=cuda.db --op_type=gemm_crc_bias --op_keys="4x5x6", or
    --gen_one_75_entry_from_80 --db=cuda.db --op_type=bmm_crc --op_keys="3x4x5x6"
    --gen_one_75_entry_from_80 --db=cuda.db --op_type=group_gemm_crc --op_keys="1x2x3;4x5x6"
  * query an sm75 entry for the given op_type and op_keys:
    --query_device=75 --db=cuda.db --op_type=gemm_crc_bias --op_keys="4x5x6"
  * query entries for all devices for the given op_type and op_keys:
    --query_all_devices --db=cuda.db --op_type=gemm_crc_bias --op_keys="4x5x6"
  * query missing sm75 entries for all op_type:
    --db=cuda.db --op_type="all" --query_missing_75_entries_from_80
  * generate missing sm75 entries for gemm_crc_bias:
    --db=cuda.db --op_type="gemm_crc_bias" --gen_75_entries_from_80
""",
    )

    arg_parser.add_argument("--db", required=True, help="specify the data base")
    arg_parser.add_argument(
        "--table", default="cuda_gemm", choices=["cuda_gemm"], help="specify the table"
    )
    arg_parser.add_argument(
        "--op_type", required=True, help="specify the op type of the cache entry"
    )
    arg_parser.add_argument(
        "--op_keys",
        help="specify the keys that will be used to construct exec_hash_sh1 for "
        "the cache entry. Each op_type follows its own format below: "
        "  * gemm family: MxNxK, e.g. --op_keys='8x4x2'"
        "  * bmm family: BxMxNxK, e.g., --op_keys='4x8x4x2'"
        "  * group_gemm family: BxMxNxK, e.g., --op_keys='8x4x2;4x5x2;3x4x5' "
        "        where gemm's problem sizes are separated by semicolon",
    )
    arg_parser.add_argument(
        "--query_all_devices",
        action="store_true",
        help="query cache entries for all devices in the current table",
    )
    arg_parser.add_argument(
        "--query_device",
        choices=["75", "80"],
        help="query cache entries for a single device in the current table",
    )
    arg_parser.add_argument(
        "--gen_one_75_entry_from_80",
        action="store_true",
        help="generate a cuda sm75 entry based on an existing sm80 entry. "
        "If --algo is not specified, we will generate an algo field based "
        "the existing algo field in the sm80 entry. We make our best guess "
        "to ensure the auto-generated algo would work for sm75. If it did "
        "not work, you could use --algo to specify a correct one.",
    )
    arg_parser.add_argument(
        "--gen_75_entries_from_80",
        action="store_true",
        help="generate a missing cuda sm75 entry for each existing sm80 entry "
        "for the given op_type. If op_type is 'all', we will generate a missing "
        "cuda sm75 entry from each existing sm80 entry for all op_type(s).",
    )
    arg_parser.add_argument(
        "--query_missing_75_entries_from_80",
        action="store_true",
        help="for the given op_type, print out all missing cuda sm75 entries if "
        "there exists an sm80 counterpart. If op_type is 'all', we print all "
        "missing cuda sm75 entries for all op_type(s).",
    )
    arg_parser.add_argument(
        "--algo",
        help="specify the algo (e.g. cutlass_tensorop_f16_xxx_tn_align_1_8) for "
        "the entry. We will set the algo field to this passed argument for "
        "the generated entry.",
    )
    arg_parser.add_argument(
        "--del_entry_by_id", help="delete an entry with the specified id"
    )

    return arg_parser.parse_args()


def main():
    """The main entry of the tool

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    args = parse_cmd_arguments()

    db_file = os.path.abspath(args.db)
    if not os.path.exists(db_file):
        raise RuntimeError("database does not exist at: " + db_file)

    db_conn = sqlite3.connect(db_file)

    if args.del_entry_by_id is not None:
        del_entry(
            db_conn=db_conn,
            table=args.table,
            entry_id=args.del_entry_by_id,
            op_type=args.op_type,
        )
        return

    if args.query_missing_75_entries_from_80:
        process_missing_75_entries_from_80(
            db_conn=db_conn,
            table=args.table,
            op_type=args.op_type,
            gen_sm75_entry=False,
        )
        return

    if args.gen_75_entries_from_80:
        process_missing_75_entries_from_80(
            db_conn=db_conn,
            table=args.table,
            op_type=args.op_type,
            gen_sm75_entry=True,
        )
        return

    if args.query_device is not None or args.query_all_devices:
        query_device = None if args.query_all_devices else args.query_device
        assert args.op_type != "all", "cannot query 'all' op_type"
        db_conn_cur = db_conn.cursor()
        query_cache(
            db_conn_cur=db_conn_cur,
            table=args.table,
            op_type=args.op_type,
            op_keys=args.op_keys,
            query_template=DEFAULT_QUERY_TEMPLATE,
            algo=args.algo,
            device=query_device,
        )
        return

    if args.op_keys is None:
        raise RuntimeError("Please specify op_keys")

    if args.gen_one_75_entry_from_80:
        gen_one_75_entry_from_80(
            db_conn=db_conn,
            table=args.table,
            op_type=args.op_type,
            op_keys=args.op_keys,
            algo=args.algo,
        )
        return


if __name__ == "__main__":
    main()
