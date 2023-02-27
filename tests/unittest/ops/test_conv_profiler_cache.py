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
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from aitemplate.backend.profiler_cache import ProfileCacheDB

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target


_LOGGER = logging.getLogger(__name__)


class ConvProfilerCacheTestCase(unittest.TestCase):
    def _test(
        self,
        first_dim,
        logger,
        test_name="conv2d",
    ):
        target = detect_target()

        X = Tensor(
            shape=[first_dim, 28, 28, 128],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 128],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        OP = ops.conv2d(stride=1, pad=1, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        with self.assertLogs(
            logger=logger,
            level="INFO",
        ) as logs:
            compile_model(
                Y,
                target,
                "./tmp",
                test_name,
                dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
            )

        return "\n".join(logs.output)

    def _run_test(
        self,
        first_dim,
        test_name,
        logger,
        cache_dir,
    ):
        old_trick = os.environ.get("TRICK_CI_ENV", None)
        old_cache = os.environ.get("CACHE_DIR", None)
        try:
            os.environ["TRICK_CI_ENV"] = "1"
            os.environ["CACHE_DIR"] = f"{cache_dir}/{test_name}"
            return self._test(
                first_dim=first_dim,
                logger=logger,
                test_name=test_name,
            )
        finally:
            if old_trick is not None:
                os.environ["TRICK_CI_ENV"] = old_trick
            else:
                os.environ.pop("TRICK_CI_ENV")
            if old_cache is not None:
                os.environ["CACHE_DIR"] = old_cache
            else:
                os.environ.pop("CACHE_DIR")

    def test_conv_profiler_cache(self):
        first_dim = IntImm(4)
        test_name = "conv2d_profiler_cache"
        logger = "aitemplate.compiler.transform.profile"

        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            run1_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 1 profilers", run1_logs)

            run2_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 0 profilers", run2_logs)

    def test_conv_profiler_cache_versioning(self):
        first_dim = IntImm(4)
        test_name = "conv2d_profiler_cache_versioning"
        logger = "aitemplate.backend.profiler_cache"
        cache_version_property = "conv_cache_version"
        target_name = detect_target().name()

        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=1,  # version
            ):
                run1_before_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_conv_1' does not exist in the db",
                    run1_before_version_change_logs,
                )

                run2_before_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_conv_1' exists in the db",
                    run2_before_version_change_logs,
                )

            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=2,  # version
            ):
                run1_after_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_conv_2' does not exist in the db",
                    run1_after_version_change_logs,
                )

                run2_after_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_conv_2' exists in the db",
                    run2_after_version_change_logs,
                )

    def test_conv_profiler_force_cache(self):
        first_dim = IntImm(4)
        test_name = "conv2d_profiler_force_cache"
        cache_version_property = "conv_cache_version"

        old_force_cache = os.environ.get("AIT_FORCE_PROFILER_CACHE", None)
        logger = "aitemplate.backend.profiler_cache"
        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=1,  # version
            ):
                _LOGGER.info("force cache with no cache 1")
                os.environ["AIT_FORCE_PROFILER_CACHE"] = "1"
                with self.assertRaisesRegex(
                    RuntimeError, "force_cache is enabled but we could not find"
                ):
                    self._run_test(
                        first_dim=first_dim,
                        test_name=test_name,
                        logger=logger,
                        cache_dir=tmp_dirname,
                    )

                del os.environ["AIT_FORCE_PROFILER_CACHE"]
                _LOGGER.info("make cache 1")
                self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )

                os.environ["AIT_FORCE_PROFILER_CACHE"] = "1"
                _LOGGER.info("force cache with no cache 1")
                self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )

        if old_force_cache is not None:
            os.environ["AIT_FORCE_PROFILER_CACHE"] = old_force_cache
        else:
            del os.environ["AIT_FORCE_PROFILER_CACHE"]

    def test_conv_profiler_cache_dynamic(self):
        first_dim = IntVar([2, 8])
        test_name = "conv2d_profiler_cache_dynamic"
        logger = "aitemplate.compiler.transform.profile"

        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            run1_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 1 profilers", run1_logs)

            run2_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 1 profilers", run2_logs)


if __name__ == "__main__":
    unittest.main()
