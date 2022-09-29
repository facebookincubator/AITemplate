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
SQLite backend for conv/gemm profiling cache
"""
import enum
import sqlite3

from typing import Any, Dict, Tuple

import jinja2

from ..utils import logger

# pylint: disable=W0613


class CacheMode(enum.Enum):
    r"""Enum for cache mode

    Profiling cache can be stored locally or remotely.
    For LOCAL mode, the cache is stored in a SQLite database.
    For REMOTE mode, the profiled results can be queried with RESTFul API.

    REMOTE mode is not implemented yet.
    """

    LOCAL = 1
    REMOTE = 2


GEMM_INIT_TEMPLATE = jinja2.Template(
    """
 CREATE TABLE IF NOT EXISTS {{dev}}_gemm (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192) NOT NULL,
  exec_entry_sha1 VARCHAR(64) NOT NULL,
  dtype_a INTEGER NOT NULL,
  dtype_b INTEGER NOT NULL,
  dtype_c INTEGER NOT NULL,
  dtype_acc INTEGER NOT NULL,
  major_a INTEGER NOT NULL,
  major_b INTEGER NOT NULL,
  major_c INTEGER NOT NULL,
  op_type VARCHAR(512) NOT NULL,
  epilogue VARCHAR(512) NOT NULL,
  device VARCHAR(16) NOT NULL,
  algo VARCHAR(512) NOT NULL,
  workspace INTEGER DEFAULT 0,
  duration FLOAT DEFAULT -1,
  split_k INTEGER DEFAULT 1,
  pshape VARCHAR(64) NOT NULL,
  template_ver INTEGER NOT NULL DEFAULT 290,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
);
"""
)

GEMM_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT algo, workspace, split_k
FROM {{dev}}_gemm
WHERE
dtype_a={{dtype_a}} AND
dtype_b={{dtype_b}} AND
dtype_c={{dtype_c}} AND
dtype_acc={{dtype_acc}} AND
major_a={{major_a}} AND
major_b={{major_b}} AND
major_c={{major_c}} AND
op_type='{{op_type}}' AND
device='{{device}}' AND
epilogue={{epilogue}} AND
pshape='{{pshape}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
"""
)

GEMM_INSERT_TEMPLATE = jinja2.Template(
    """
INSERT INTO {{dev}}_gemm (
    exec_entry,
    exec_entry_sha1,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    major_a,
    major_b,
    major_c,
    op_type,
    epilogue,
    device,
    algo,
    workspace,
    split_k,
    pshape
)
VALUES (
    '{{exec_entry}}',
    '{{exec_entry_sha1}}',
    {{dtype_a}},
    {{dtype_b}},
    {{dtype_c}},
    {{dtype_acc}},
    {{major_a}},
    {{major_b}},
    {{major_c}},
    '{{op_type}}',
    {{epilogue}},
    '{{device}}',
    '{{algo}}',
    {{workspace}},
    {{split_k}},
    '{{pshape}}'
);
"""
)

GEMM_ENTRY_QUERY = jinja2.Template(
    """
SELECT id
FROM {{dev}}_gemm
WHERE
op_type='{{op_type}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
"""
)

CONV_INIT_TEMPLATE = jinja2.Template(
    """
 CREATE TABLE IF NOT EXISTS {{dev}}_conv (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192) NOT NULL,
  exec_entry_sha1 VARCHAR(64) NOT NULL,
  dtype_a INTEGER NOT NULL,
  dtype_b INTEGER NOT NULL,
  dtype_c INTEGER NOT NULL,
  dtype_acc INTEGER NOT NULL,
  major_a INTEGER NOT NULL,
  major_b INTEGER NOT NULL,
  major_c INTEGER NOT NULL,
  kh INTEGER NOT NULL,
  kw INTEGER NOT NULL,
  co INTEGER NOT NULL,
  stride INTEGER NOT NULL,
  pad INTEGER NOT NULL,
  dilate INTEGER NOT NULL,
  op_type VARCHAR(512) NOT NULL,
  epilogue VARCHAR(512) NOT NULL,
  device VARCHAR(16) NOT NULL,
  algo VARCHAR(512) NOT NULL,
  workspace INTEGER DEFAULT 0,
  duration FLOAT DEFAULT -1,
  split_k INTEGER DEFAULT 1,
  template_ver INTEGER NOT NULL DEFAULT 290,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
);
"""
)

CONV_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT algo, workspace
FROM {{dev}}_conv
WHERE
dtype_a={{dtype_a}} AND
dtype_b={{dtype_b}} AND
dtype_c={{dtype_c}} AND
dtype_acc={{dtype_acc}} AND
major_a={{major_a}} AND
major_b={{major_b}} AND
major_c={{major_c}} AND
kh={{kh}} AND
kw={{kw}} AND
co={{co}} AND
stride={{stride}} AND
pad={{pad}} AND
dilate={{dilate}} AND
op_type='{{op_type}}' AND
device='{{device}}' AND
epilogue={{epilogue}} AND
split_k={{split_k}} AND
exec_entry_sha1='{{exec_entry_sha1}}';
"""
)

CONV_INSERT_TEMPLATE = jinja2.Template(
    """
INSERT INTO {{dev}}_conv (
    exec_entry,
    exec_entry_sha1,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    major_a,
    major_b,
    major_c,
    kh,
    kw,
    co,
    stride,
    pad,
    dilate,
    op_type,
    epilogue,
    device,
    algo,
    workspace,
    split_k
)
VALUES (
    '{{exec_entry}}',
    '{{exec_entry_sha1}}',
    {{dtype_a}},
    {{dtype_b}},
    {{dtype_c}},
    {{dtype_acc}},
    {{major_a}},
    {{major_b}},
    {{major_c}},
    {{kh}},
    {{kw}},
    {{co}},
    {{stride}},
    {{pad}},
    {{dilate}},
    '{{op_type}}',
    {{epilogue}},
    '{{device}}',
    '{{algo}}',
    {{workspace}},
    {{split_k}}
);
"""
)


NORM_INIT_TEMPLATE = jinja2.Template(
    """
 CREATE TABLE IF NOT EXISTS {{dev}}_normalization (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_entry VARCHAR(8192) NOT NULL,
  exec_entry_sha1 VARCHAR(64) NOT NULL,
  dtype_in INTEGER NOT NULL,
  dtype_acc INTEGER NOT NULL,
  dtype_out INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  op_type VARCHAR(512) NOT NULL,
  device VARCHAR(16) NOT NULL,
  algo VARCHAR(512) NOT NULL,
  workspace INTEGER DEFAULT 0,
  duration FLOAT DEFAULT -1,
  template_ver INTEGER NOT NULL DEFAULT 290,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
);
"""
)

NORM_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT algo, workspace
FROM {{dev}}_normalization
WHERE
dtype_in={{dtype_in}} AND
dtype_out={{dtype_out}} AND
dtype_acc={{dtype_acc}} AND
rank={{rank}} AND
op_type='{{op_type}}' AND
device='{{device}}' AND
exec_entry_sha1='{{exec_entry_sha1}}';
"""
)

NORM_INSERT_TEMPLATE = jinja2.Template(
    """
INSERT INTO {{dev}}_normalization (
    exec_entry,
    exec_entry_sha1,
    dtype_in,
    dtype_out,
    dtype_acc,
    rank,
    op_type,
    device,
    algo,
    workspace
)
VALUES (
    '{{exec_entry}}',
    '{{exec_entry_sha1}}',
    {{dtype_in}},
    {{dtype_out}},
    {{dtype_acc}},
    {{rank}},
    '{{op_type}}',
    '{{device}}',
    '{{algo}}',
    {{workspace}}
);
"""
)


class ProfileCacheDB(object):
    r"""Local SQLite profile cache database."""

    def __init__(
        self, target: str, path: str = None, uri: str = None, port: str = None
    ):
        r"""=

        Parameters
        ----------
        target : str
            target device name. CUDA or ROCM
        path : str, optional
            path to the database file. If not specified, a temporary file is created.
        uri : str, optional
            uri to the RESFul API (Not implemented yet)
        port : str, optional
            port to the RESFul API (Not implemented yet)

        """
        self._target = target
        self._mode = CacheMode.LOCAL
        self._db_commit_flag = False
        if uri is not None:
            self._mode = CacheMode.REMOTE
        if self._mode == CacheMode.LOCAL:
            assert path is not None
            self._con = sqlite3.connect(path)
            self._cur = self._con.cursor()
            self._init_db()
        else:
            raise NotImplementedError

    def _init_db(self):
        """Creates table in cache."""
        self._create_gemm_table()
        self._create_conv_table()
        self._create_norm_table()

    def _create_gemm_table(self):
        """Creates gemm table."""
        sql = GEMM_INIT_TEMPLATE.render(dev=self._target)
        self._cur.execute(sql)
        self._con.commit()

    def _create_conv_table(self):
        """Creates conv table."""
        sql = CONV_INIT_TEMPLATE.render(dev=self._target)
        self._cur.execute(sql)
        self._con.commit()

    def _create_norm_table(self):
        """Creates conv table."""
        sql = NORM_INIT_TEMPLATE.render(dev=self._target)
        self._cur.execute(sql)
        self._con.commit()

    def _query(self, sql: str) -> Tuple[str, int]:
        """a function to query op from cache

        Parameters
        ----------
        sql : str
            sql statement for query

        Returns
        -------
        Tuple
            profiling results
        """
        if self._mode == CacheMode.LOCAL:
            if self._db_commit_flag:
                self._con.commit()
                self._db_commit_flag = False
            self._cur.execute(sql)
            out = self._cur.fetchall()
            if len(out) == 0:
                return None
            return out[0]

        raise NotImplementedError

    def query_gemm(self, args: Dict[str, Any]) -> Tuple[str, int]:
        """a function to query gemm op epilogue from cache

        Parameters
        ----------
        args : Dict
            gemm query entry

        Returns
        -------
        Tuple
            profiling results
        """
        sql = GEMM_QUERY_TEMPLATE.render(dev=self._target, **args)
        return self._query(sql)

    def query_conv(self, args: Dict[str, Any]) -> Tuple[str, int]:
        """a function to query conv op epilogue from cache,
        here we use the same sql table for conv and gemm

        Parameters
        ----------
        args : Dict
            Conv query entry

        Returns
        -------
        Tuple
            profiling results
        """
        sql = CONV_QUERY_TEMPLATE.render(dev=self._target, **args)
        return self._query(sql)

    def query_normalization(self, args: Dict[str, Any]) -> Tuple[str, int]:
        """a function to query normalization op epilogue from cache

        Parameters
        ----------
        args : Dict
            Conv query entry

        Returns
        -------
        Tuple
            profiling results
        """
        sql = NORM_QUERY_TEMPLATE.render(dev=self._target, **args)
        return self._query(sql)

    def _insert(self, query_sql: str, insert_sql: str) -> None:
        """a function to insert op into cache

        Parameters
        ----------
        query_sql : str
            cache query sql
        insert_sql: str
            cache insert sql
        """
        if self._mode == CacheMode.LOCAL:
            self._cur.execute(query_sql)
            out = self._cur.fetchall()
            if len(out) == 0:
                self._cur.execute(insert_sql)
                self._db_commit_flag = True
            else:
                logger.info(__name__, "Ignore repeat profile_record: " + query_sql)

    def insert_gemm(self, args: Dict[str, Any]) -> None:
        """a function to insert gemm op epilogue into cache

        Parameters
        ----------
        args : Dict
            Gemm Record Entry
        """
        query_sql = GEMM_QUERY_TEMPLATE.render(
            dev=self._target,
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
            pshape=args["pshape"],
            exec_entry_sha1=args["exec_entry_sha1"],
        )
        insert_sql = GEMM_INSERT_TEMPLATE.render(dev=self._target, **args)
        self._insert(query_sql, insert_sql)

    def insert_conv(self, args: Dict[str, Any]) -> None:
        """a function to insert conv op epilogue into cache,
        here we use the same sql table for conv and gemm

        Parameters
        ----------
        args : Dict
            Conv Record Entry

        """
        query_sql = CONV_QUERY_TEMPLATE.render(
            dev=self._target,
            dtype_a=args["dtype_a"],
            dtype_b=args["dtype_b"],
            dtype_c=args["dtype_c"],
            dtype_acc=args["dtype_acc"],
            major_a=args["major_a"],
            major_b=args["major_b"],
            major_c=args["major_c"],
            kh=args["kh"],
            kw=args["kw"],
            co=args["co"],
            stride=args["stride"],
            pad=args["pad"],
            dilate=args["dilate"],
            op_type=args["op_type"],
            device=args["device"],
            epilogue=args["epilogue"],
            split_k=args["split_k"],
            exec_entry_sha1=args["exec_entry_sha1"],
        )
        insert_sql = CONV_INSERT_TEMPLATE.render(dev=self._target, **args)
        self._insert(query_sql, insert_sql)

    def insert_normalization(self, args: Dict[str, Any]) -> None:
        """a function to insert conv op epilogue into cache,
        here we use the same sql table for conv and gemm

        Parameters
        ----------
        args : Dict
            Conv Record Entry

        """
        query_sql = NORM_QUERY_TEMPLATE.render(
            dev=self._target,
            dtype_in=args["dtype_in"],
            dtype_acc=args["dtype_acc"],
            dtype_out=args["dtype_out"],
            rank=args["rank"],
            op_type=args["op_type"],
            device=args["device"],
            exec_entry_sha1=args["exec_entry_sha1"],
        )
        insert_sql = NORM_INSERT_TEMPLATE.render(dev=self._target, **args)
        self._insert(query_sql, insert_sql)

    def __del__(self):
        self._con.commit()
        self._con.close()
