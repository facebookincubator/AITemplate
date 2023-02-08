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
import logging
import sqlite3

from typing import Any, Dict, Tuple

import jinja2

# pylint: disable=W0613


_LOGGER = logging.getLogger(__name__)


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
 CREATE TABLE IF NOT EXISTS {{dev}}_gemm_{{version}} (
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
FROM {{dev}}_gemm_{{version}}
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
INSERT INTO {{dev}}_gemm_{{version}} (
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

CONV_INIT_TEMPLATE = jinja2.Template(
    """
 CREATE TABLE IF NOT EXISTS {{dev}}_conv_{{version}} (
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
FROM {{dev}}_conv_{{version}}
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
INSERT INTO {{dev}}_conv_{{version}} (
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

CONV3D_INIT_TEMPLATE = jinja2.Template(
    """
 CREATE TABLE IF NOT EXISTS {{dev}}_conv3d_{{version}} (
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
  kd INTEGER NOT NULL,
  kh INTEGER NOT NULL,
  kw INTEGER NOT NULL,
  co INTEGER NOT NULL,
  stride_d INTEGER NOT NULL,
  stride_h INTEGER NOT NULL,
  stride_w INTEGER NOT NULL,
  pad_d INTEGER NOT NULL,
  pad_h INTEGER NOT NULL,
  pad_w INTEGER NOT NULL,
  dilate_d INTEGER NOT NULL,
  dilate_h INTEGER NOT NULL,
  dilate_w INTEGER NOT NULL,
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

CONV3D_QUERY_TEMPLATE = jinja2.Template(
    """
SELECT algo, workspace
FROM {{dev}}_conv3d_{{version}}
WHERE
dtype_a={{dtype_a}} AND
dtype_b={{dtype_b}} AND
dtype_c={{dtype_c}} AND
dtype_acc={{dtype_acc}} AND
major_a={{major_a}} AND
major_b={{major_b}} AND
major_c={{major_c}} AND
kd={{kd}} AND
kh={{kh}} AND
kw={{kw}} AND
co={{co}} AND
stride_d={{stride_d}} AND
stride_h={{stride_h}} AND
stride_w={{stride_w}} AND
pad_d={{pad_d}} AND
pad_h={{pad_h}} AND
pad_w={{pad_w}} AND
dilate_d={{dilate_d}} AND
dilate_h={{dilate_h}} AND
dilate_w={{dilate_w}} AND
op_type='{{op_type}}' AND
device='{{device}}' AND
epilogue={{epilogue}} AND
split_k={{split_k}} AND
exec_entry_sha1='{{exec_entry_sha1}}';
"""
)

CONV3D_INSERT_TEMPLATE = jinja2.Template(
    """
INSERT INTO {{dev}}_conv3d_{{version}} (
    exec_entry,
    exec_entry_sha1,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    major_a,
    major_b,
    major_c,
    kd,
    kh,
    kw,
    co,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dilate_d,
    dilate_h,
    dilate_w,
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
    {{kd}},
    {{kh}},
    {{kw}},
    {{co}},
    {{stride_d}},
    {{stride_h}},
    {{stride_w}},
    {{pad_d}},
    {{pad_h}},
    {{pad_w}},
    {{dilate_d}},
    {{dilate_h}},
    {{dilate_w}},
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


CHECK_TABLE_EXISTENCE_TEMPLATE = jinja2.Template(
    """
SELECT name FROM sqlite_master WHERE type='table' AND name='{{table_name}}';
"""
)


QUERY_ALL_TABLES_TEMPLATE = jinja2.Template(
    """
SELECT name FROM sqlite_master WHERE type='table';
"""
)


__AIT_CACHE_VERSION__ = 2


def ait_cache_version() -> int:
    return __AIT_CACHE_VERSION__


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
        # Some design rationales:
        #   * All tables share the version number, because we are exposing the
        #     the cache version. Using a single version number seems to make it
        #     more clean. One caveat is that we are going to re-create all tables
        #     even if some of them are not involved with the breaking changes.
        #     It seems to be fine as we expect the frequency of cache version
        #     updated to be quite low.
        #   * We only keep a single table (i.e. version) for each category (
        #     gemm, conv and norm) to simplify how we handle breaking changes
        #     and rollbacks caused by failures in the updated version.
        #     For example, if we keep multiple versions (i.e. tables) for gemm,
        #     we would have to consider how we were going to maintain those versions.
        #     We could choose the old working version upon rollback, but we might
        #     leave some content from the failing version in the db. How are we
        #     going to update the db if we update the version again, and so on.
        # TODO: add similar version control for norm
        self._gemm_cache_version = ait_cache_version()
        self._conv_cache_version = ait_cache_version()
        self._conv3d_cache_version = ait_cache_version()
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
        self._create_conv3d_table()
        self._create_norm_table()

    @property
    def gemm_cache_version(self) -> int:
        return self._gemm_cache_version

    @property
    def conv_cache_version(self) -> int:
        return self._conv_cache_version

    @property
    def conv3d_cache_version(self) -> int:
        return self._conv3d_cache_version

    def _create_gemm_table(self):
        """Creates gemm table."""
        version = self.gemm_cache_version
        if not self._table_exists("gemm", version):
            _LOGGER.info(
                "Temporarily keeping the old gemm cache versions if exist",
            )
            # FIXME: will delete unmatched version once we get into production
            # self._delete_existing_table("gemm")

            _LOGGER.info(
                f"Creating a new gemm table with {version=}",
            )
            sql = GEMM_INIT_TEMPLATE.render(
                dev=self._target,
                version=version,
            )
            self._cur.execute(sql)
            self._con.commit()

    def _create_conv_table(self):
        """Creates conv table."""
        version = self.conv_cache_version
        if not self._table_exists("conv", version):
            _LOGGER.info(
                "Temporarily keeping the old conv cache versions if exist",
            )
            # FIXME: will delete unmatched version once we get into production
            # self._delete_existing_table("conv")

            _LOGGER.info(
                f"Creating a new conv table with {version=}",
            )
            sql = CONV_INIT_TEMPLATE.render(
                dev=self._target,
                version=version,
            )
            self._cur.execute(sql)
            self._con.commit()

    def _create_conv3d_table(self):
        """Creates conv3d table."""
        version = self.conv3d_cache_version
        if not self._table_exists("conv3d", version):
            _LOGGER.info(
                "Temporarily keeping the old conv3d cache versions if exist",
            )
            # FIXME: will delete unmatched version once we get into production
            # self._delete_existing_table("conv3d")

            _LOGGER.info(
                f"Creating a new conv3d table with {version=}",
            )
            sql = CONV3D_INIT_TEMPLATE.render(
                dev=self._target,
                version=version,
            )
            self._cur.execute(sql)
            self._con.commit()

    def _create_norm_table(self):
        """Creates conv table."""
        sql = NORM_INIT_TEMPLATE.render(dev=self._target)
        self._cur.execute(sql)
        self._con.commit()

    def _table_exists(self, table_kind, cache_version):
        """Check if the table of given kind and cache version exists."""
        table_name = f"{self._target}_{table_kind}_{cache_version}"
        sql = CHECK_TABLE_EXISTENCE_TEMPLATE.render(table_name=table_name)
        self._cur.execute(sql)
        tables = self._cur.fetchall()

        if tables:
            _LOGGER.info(
                f"{table_name=} exists in the db",
            )
            return True
        else:
            _LOGGER.info(
                f"{table_name=} does not exist in the db, possible version mismatch!",
            )
            return False

    def _delete_existing_table(self, table_kind):
        """Delete an existing table in the db"""
        sql = QUERY_ALL_TABLES_TEMPLATE.render()
        self._cur.execute(sql)
        all_tables = self._cur.fetchall()
        if len(all_tables) == 0:
            _LOGGER.info("deleting table: skip empty table")
            return

        target_tables = [
            table[0]
            for table in all_tables
            if table[0].startswith(f"{self._target}_{table_kind}")
        ]
        assert len(target_tables) != 0, f"no {table_kind} table exists"
        # To simplify the logic, we only keep a single table for each kind
        assert (
            len(target_tables) == 1
        ), f"expected only one {table_kind} table but got {target_tables=}"
        _LOGGER.info(f"deleting table {target_tables[0]=}")
        self._cur.execute(f"DROP TABLE {target_tables[0]}")

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
        sql = GEMM_QUERY_TEMPLATE.render(
            dev=self._target,
            version=self.gemm_cache_version,
            **args,
        )
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
        sql = CONV_QUERY_TEMPLATE.render(
            dev=self._target,
            version=self.conv_cache_version,
            **args,
        )
        return self._query(sql)

    def query_conv3d(self, args: Dict[str, Any]) -> Tuple[str, int]:
        """a function to query conv op epilogue from cache,
        here we use the same sql table for conv and gemm

        Parameters
        ----------
        args : Dict
            Conv3d query entry

        Returns
        -------
        Tuple
            profiling results
        """
        sql = CONV3D_QUERY_TEMPLATE.render(
            dev=self._target,
            version=self.conv3d_cache_version,
            **args,
        )
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
                _LOGGER.info("Ignore repeat profile_record: " + query_sql)

    def insert_gemm(self, args: Dict[str, Any]) -> None:
        """a function to insert gemm op epilogue into cache

        Parameters
        ----------
        args : Dict
            Gemm Record Entry
        """
        query_sql = GEMM_QUERY_TEMPLATE.render(
            dev=self._target,
            version=self.gemm_cache_version,
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
        insert_sql = GEMM_INSERT_TEMPLATE.render(
            dev=self._target,
            version=self.gemm_cache_version,
            **args,
        )
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
            version=self.conv_cache_version,
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
        insert_sql = CONV_INSERT_TEMPLATE.render(
            dev=self._target,
            version=self.conv_cache_version,
            **args,
        )
        self._insert(query_sql, insert_sql)

    def insert_conv3d(self, args: Dict[str, Any]) -> None:
        """a function to insert conv op epilogue into cache,
        here we use the same sql table for conv and gemm

        Parameters
        ----------
        args : Dict
            Conv Record Entry

        """
        query_sql = CONV3D_QUERY_TEMPLATE.render(
            dev=self._target,
            version=self.conv3d_cache_version,
            dtype_a=args["dtype_a"],
            dtype_b=args["dtype_b"],
            dtype_c=args["dtype_c"],
            dtype_acc=args["dtype_acc"],
            major_a=args["major_a"],
            major_b=args["major_b"],
            major_c=args["major_c"],
            kd=args["kd"],
            kh=args["kh"],
            kw=args["kw"],
            co=args["co"],
            stride_d=args["stride_d"],
            stride_h=args["stride_h"],
            stride_w=args["stride_w"],
            pad_d=args["pad_d"],
            pad_h=args["pad_h"],
            pad_w=args["pad_w"],
            dilate_d=args["dilate_d"],
            dilate_h=args["dilate_h"],
            dilate_w=args["dilate_w"],
            op_type=args["op_type"],
            device=args["device"],
            epilogue=args["epilogue"],
            split_k=args["split_k"],
            exec_entry_sha1=args["exec_entry_sha1"],
        )
        insert_sql = CONV3D_INSERT_TEMPLATE.render(
            dev=self._target,
            version=self.conv3d_cache_version,
            **args,
        )
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
