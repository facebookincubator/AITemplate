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
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from aitemplate.backend.build_cache_base import (
    create_dir_hash,
    FileBasedBuildCache,
    is_source,
    makefile_normalizer,
    SkipBuildCache,
)

from aitemplate.backend.cuda.target_def import FBCUDA
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.debug_settings import AITDebugSettings
from aitemplate.utils.io import file_age

_LOGGER = logging.getLogger(__name__)


class BuildCacheTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _create_model_graph(self):
        dtype = "float32"
        X1 = Tensor(
            shape=[IntImm(1), IntImm(10)],
            dtype=dtype,
            name="X1",
            is_input=True,
        )
        Y = ops.expand()(X1, shape=(10, 10))
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        return Y

    def _create_model_graph2(self):
        dtype = "float32"
        Z1 = Tensor(
            shape=[IntImm(10), IntImm(1)],
            dtype=dtype,
            name="Z1",
            is_input=True,
        )
        Y = ops.expand()(Z1, shape=(10, 10))
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        return Y

    def test_file_build_cache(self):
        with patch(
            "aitemplate.backend.build_cache_base.should_skip_build_cache"
        ) as should_skip_build_cache_mock:
            should_skip_build_cache_mock.return_value = False
            with tempfile.TemporaryDirectory() as parent_dir:
                cache_dir = os.path.join(parent_dir, "build_cache")
                shutil.rmtree(cache_dir, ignore_errors=True)
                cache = FileBasedBuildCache(
                    cache_dir,
                    lru_retention_hours=0,
                    cleanup_max_age_seconds=1000,
                    debug=True,
                )
                cache.maybe_cleanup()
                assert os.path.exists(cache_dir + "/.last_cleaned")
                assert (
                    file_age(cache_dir + "/.last_cleaned") < 10.0
                ), "Last clean time should  than 10 seconds"

                build_dir_1 = os.path.join(parent_dir, "build_1")
                build_dir_2 = os.path.join(parent_dir, "build_2")

                os.makedirs(build_dir_1, exist_ok=False)
                os.makedirs(build_dir_2, exist_ok=False)
                for build_dir in [build_dir_1, build_dir_2]:
                    bp = Path(build_dir)
                    (bp / "Makefile").write_text("test.exe: test.cu")
                    (bp / "test.cu").write_text("printf('Hello, World!');")
                assert create_dir_hash(
                    [f"make {build_dir_1}"], build_dir_1
                ) == create_dir_hash([f"make {build_dir_2}"], build_dir_2)
                found_entry1, cache_key1 = cache.retrieve_build_cache(
                    [f"make {build_dir_1}"], build_dir_1
                )
                found_entry2, cache_key2 = cache.retrieve_build_cache(
                    [f"make {build_dir_2}"], build_dir_2
                )
                assert not found_entry1
                assert not found_entry2
                assert cache_key1 == cache_key2
                assert cache_key1 == create_dir_hash(
                    [f"make {build_dir_1}"], build_dir_1
                )
                (Path(build_dir_2) / "test.so").write_bytes("ELF1234".encode("ascii"))
                cache.store_build_cache(
                    [f"make {build_dir_2}"], build_dir_2, cache_key2
                )
                assert os.path.exists(os.path.join(cache_dir, cache_key2))
                found_entry1, cache_key1 = cache.retrieve_build_cache(
                    [f"make {build_dir_1}"], build_dir_1
                )
                assert os.path.exists(os.path.join(build_dir_1, "test.so"))
                assert (
                    Path(os.path.join(build_dir_1, "test.so")).read_bytes()
                    == Path(os.path.join(build_dir_2, "test.so")).read_bytes()
                )

    def test_deterministic_codegen(self, dtype="float32"):
        with SkipBuildCache():
            # Tests, whether repeated invocation of compilation results in identical generated source files
            test_name = "test_deterministic_codegen"
            basepath = "./tmp"

            # Clean previous test results. These are usually kept for debugging purposes
            # but we need a clean slate here.
            if os.path.exists(basepath):
                existing_dirs = [
                    d
                    for d in os.listdir(basepath)
                    if d.startswith(test_name)
                    and os.path.isdir(os.path.join(basepath, d))
                ]
                for d in existing_dirs:
                    oldpath = os.path.join(basepath, d)
                    if os.path.exists(oldpath) and test_name in oldpath:
                        shutil.rmtree(oldpath)
            else:
                os.mkdir(basepath)

            Y = self._create_model_graph()
            target = detect_target()
            debug_settings = AITDebugSettings(gen_standalone=False)
            dll_name = "test.so"
            build_dir = os.path.join("./tmp", test_name)
            compile_model(
                Y,
                target,
                "./tmp",
                test_name + "_1",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            hash1 = create_dir_hash(
                ["test_name"], build_dir + "_1", is_source, debug=True
            )
            cache_key_log_1 = (Path(build_dir + "_1") / "cache_key.log").read_text()
            Y = self._create_model_graph()
            target = detect_target()
            # Variant 2: Clean build
            compile_model(
                Y,
                target,
                "./tmp",
                test_name + "_2",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            hash2 = create_dir_hash(
                ["test_name"], build_dir + "_2", is_source, debug=True
            )
            cache_key_log_2 = (Path(build_dir + "_2") / "cache_key.log").read_text()

            assert (
                hash1 == hash2
            ), f"Code generation was not deterministic. Cache key mismatch between first and second code generation pass. Hint: Debug this with the help of the debug option of function create_dir_hash(...)\nLOG 1:\n{cache_key_log_1}\n------\nLOG 1:\n{cache_key_log_2}\n-----"
            # Variant 3: Build over existing build dir
            Y = self._create_model_graph()
            target = detect_target()
            compile_model(
                Y,
                target,
                "./tmp",
                test_name + "_2",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            hash3 = create_dir_hash(
                ["test_name"], build_dir + "_2", is_source, debug=True
            )
            assert (
                hash2 == hash3
            ), "Code generation was not deterministic. Cache key mismatch between second and third code generation pass. Hint: Debug this with the help of the debug option of function create_dir_hash(...)"

            # Variant 4: Let's provoke to copy the includes again, maybe to a new path?
            Y = self._create_model_graph()
            FBCUDA.cutlass_path_ = None
            compile_model(
                Y,
                target,
                "./tmp",
                test_name + "_4",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            hash4 = create_dir_hash(
                ["test_name"], build_dir + "_4", is_source, debug=True
            )

            assert (
                hash3 == hash4
            ), "Code generation was not deterministic. Cache key mismatch between third and fourth code generation pass. Hint: Debug this with the help of the debug option of function create_dir_hash(...)"

            with open(
                os.path.join(build_dir + "_4", "Makefile"), "a", encoding="utf-8"
            ) as f:
                f.write("\n")

            hash5 = create_dir_hash(
                ["test_name"], build_dir + "_4", is_source, debug=True
            )
            assert (
                hash4 != hash5
            ), "Directory hash was not sensitive to a change in the Makefile, the hashes should be different. Hint: Debug this with the help of the debug option of function create_dir_hash"
            with open(
                os.path.join(build_dir + "_4", "anything.cu"), "w", encoding="utf-8"
            ) as f:
                f.write("// Nothing, really\n")

            hash6 = create_dir_hash(
                ["test_name"], build_dir + "_4", is_source, debug=True
            )
            assert (
                hash6 != hash5
            ), "Directory hash was not sensitive to a change in a source file, the hashes should be different. Hint: Debug this with the help of the debug option of function create_dir_hash"

            os.rename(
                os.path.join(build_dir + "_4", "anything.cu"),
                os.path.join(build_dir + "_4", "anything_.cu"),
            )
            hash7 = create_dir_hash(
                ["test_name"], build_dir + "_4", is_source, debug=True
            )
            assert (
                hash7 != hash6
            ), "Directory hash was not sensitive to a change of name of a source file, the hashes should be different. Hint: Debug this with the help of the debug option of function create_dir_hash"

            Y = self._create_model_graph()
            target = detect_target()
            debug_settings = AITDebugSettings(gen_standalone=True)
            compile_model(
                Y,
                target,
                "./tmp",
                test_name + "_8",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            hash8 = create_dir_hash(
                ["test_name"], build_dir + "_8", is_source, debug=True
            )

            assert (
                hash8 != hash1
            ), "Directory hash was not sensitive to a change of Makefile (standalone codegen) and possibly source code, the hashes should be different. Hint: Debug this with the help of the debug option of function create_dir_hash"

    def test_makefile_rewrite(self):
        tmpdir = os.path.join(tempfile.gettempdir(), f"{os.getuid()}_aitemplate_tmp")
        makefile = f"""
                TMPDIR: {tmpdir}
        """
        assert tmpdir in makefile
        rewritten_makefile = makefile_normalizer(makefile.encode("utf-8")).decode(
            "utf-8"
        )
        assert tmpdir not in rewritten_makefile
        assert "$USER" in rewritten_makefile

    def test_repeated_build_dir_usage(self):
        with tempfile.TemporaryDirectory() as tempdir:
            Y = self._create_model_graph()
            target = detect_target()
            debug_settings = AITDebugSettings(gen_standalone=False)
            dll_name = "test.so"
            build_dir = Path(tempdir) / "build_dir"
            compile_model(
                Y,
                target,
                tempdir,
                "build_dir",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            test_so_path = build_dir / "test.so"
            assert test_so_path.exists()
            test_so_content = test_so_path.read_bytes()

            # Compile a slightly different model into the same directory
            Y2 = self._create_model_graph2()
            debug_settings = AITDebugSettings(gen_standalone=False)
            compile_model(
                Y2,
                target,
                tempdir,
                "build_dir",
                dll_name=dll_name,
                debug_settings=debug_settings,
            )
            assert test_so_path.exists()
            test_so_new_content = test_so_path.read_bytes()
            self.assertNotEqual(
                test_so_content,
                test_so_new_content,
                "The test.so should have been overwritten and different. Maybe a build cache was used and did not overwrite the file properly?",
            )


if __name__ == "__main__":
    unittest.main()
