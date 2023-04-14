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

import hashlib
import logging
import os
import random
import secrets
import shutil
import tempfile

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from aitemplate.utils import environ as aitemplate_env

from aitemplate.utils.io import file_age, touch

_LOGGER = logging.getLogger(__name__)


# File extensions to be considered source files
source_extensions = {
    "cpp",
    "h",
    "cu",
    "cuh",
    "c",
    "hpp",
    "hxx",
    "py",
    "cxx",
    "cc",
    "version",
    "binhash",
    "hash",
}

source_filenames = {
    # needs to be lowercase, because everything is lowercased before comparison
    # Filenames in here are considered source files, even if their extension would
    # suggest they are cache artifacts
    "makefile"
}

source_filename_prefixes = ["makefile"]

# File extensions of files to be considered cache artifacts ( unless they are considered source files )
# note: we're not caching .obj files anymore as these are not strictly necessary to keep.
cache_extensions = {"so", "dll", "exe", ""}

skip_cache_flag = False  # Global flag that cache implementations should check whether
# the cache is enabled or not. Used by skip_build_cache decorator


class SkipBuildCache:
    def __init__(self, context_skip_cache_flag: bool = True):
        """
        Context manager to temporarily disable the build cache within an execution context.
        """
        self.context_skip_cache_flag = context_skip_cache_flag

    def __enter__(self):
        global skip_cache_flag
        self.old_skip_cache_flag = skip_cache_flag
        skip_cache_flag = self.context_skip_cache_flag

    def __exit__(self, *args, **kwargs):
        global skip_cache_flag
        skip_cache_flag = self.old_skip_cache_flag


def should_skip_build_cache():
    """
    This function should be called by cache implementations to determine whether the cache should be skipped or not
    """
    global skip_cache_flag
    if skip_cache_flag:
        return True
    skip_percentage = aitemplate_env.ait_build_cache_skip_percentage()
    if skip_percentage is not None:
        skip_percentage = int(skip_percentage)
        assert (
            skip_percentage >= 0 and skip_percentage <= 100
        ), f"Skip percentage has to be in the range [0,100]. Actual value: {skip_percentage}"
        if skip_percentage == 100:
            return True
        if skip_percentage == 0:
            return False
        rndi = random.randint(0, 99)
        if rndi < skip_percentage:
            return True
    return False


def filename_norm_split(filename: str) -> Tuple[str, str]:
    """
    Splits filename into basename and extension
    and lowercases results to enable simple lookup
    in a case-insensitive manner.

    Args:
        filename (str): Filename/Path to split

    Returns:
        Tuple[str,str]: file basename, file extension
    """
    file_basename = os.path.basename(filename).lower()
    file_parts = file_basename.split(".")
    if len(file_parts) > 1:
        file_ext = file_parts[-1]
    else:
        file_ext = ""
    return file_basename, file_ext


def is_source(filename: str) -> bool:
    """
    Simple filter function, returns true if the passed filename is considered
    to be a source file (used to build the cache key) for the purpose of caching

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a source file
    """
    file_basename, file_ext = filename_norm_split(filename)
    return (
        (file_basename in source_filenames)
        or (file_ext in source_extensions)
        or any(file_basename.startswith(p) for p in source_filename_prefixes)
    )


def is_cache_artifact(filename: str) -> bool:
    """
    Simple filter function, returns true if the passed filename is considered
    to be a cacheable artifact (not used to build cache key, but stored in cache)
    for the purpose of caching

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a cache artifact
    """
    file_basename, file_ext = filename_norm_split(filename)
    return not is_source(filename) and file_ext in cache_extensions


def is_bin_file(filename: str) -> bool:
    """
    Simple filter function, returns true if the passed filename is considered
    to be a bin file which needs to be considered for the purpose of creating
    a cache-key, but may be deleted after an initial build.

    bin files are hashed, and their hashes are kept in a small separete file
    for future use when building the cache key. So the hash is not lost, even if the binary
    file is deleted.

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a binary file in the above sense
    """
    return filename.lower().endswith(".bin")


def makefile_normalizer(makefile_content_orig: bytes) -> bytes:
    """Normalize the content of the makefile for hashing purposes (nothing else!),
    so that it can be compared to other Makefiles
    generated by different users on different systems"""
    makefile_content = makefile_content_orig.decode("utf-8")
    tmpdir = tempfile.gettempdir()
    userid = str(os.getuid())
    user_tmpdir = os.path.join(tmpdir, userid)
    makefile_content = makefile_content.replace(user_tmpdir, "/tmp/$USER")
    makefile_content = makefile_content.replace(tmpdir, "/tmp")
    return makefile_content.encode("utf-8")


def create_dir_hash(
    cmds: List[str],
    build_dir: str,
    filter_func: Callable[[str], bool] = is_source,
    debug=False,
) -> str:
    """Create a hash of the (source file) contents of a build directory, used for
    creating a cache key of an entire directory along with the build commands.

    Args:
        cmds (List[str]): Build commands to be incorporated in hash key computation
        build_dir (str): Path to build directory ( not part of hash )
        filter_func (Callable[[str], bool], optional): Filter function which determines whether a given file is considered a source file or not. Defaults to is_source(path).
        debug (bool, optional): Whether to write a 'cache_key.log' file into the build directory, so that cache misses can be debugged more easily. Defaults to False.

    Returns:
        str: SHA256 Hash of the build directory contents in the form of a hexdigest string.
    """
    hash_log = None
    try:
        if not os.path.isdir(build_dir):
            return "empty_dir"
        if debug:
            hash_log = open(  # noqa: P201 - this is actually closed properly in the finally close below
                os.path.join(build_dir, "cache_key.log"), mode="a", encoding="utf8"
            )
            hash_log.write(f"Building dir hash of {build_dir}\n")
        basepath = Path(build_dir)
        files = [p.relative_to(basepath) for p in basepath.rglob("*") if not p.is_dir()]
        hash_object = hashlib.sha256()
        for cmd in cmds:
            _cmd = cmd.replace(
                build_dir, "${BUILD_DIR}"
            )  # Make sure we can cache regardless of the build directory location.
            hash_object.update(_cmd.encode("utf-8"))
            if debug:
                hash_log.write(f"\tCOMMAND: {_cmd} -> {hash_object.hexdigest()}\n")
        for fpath in sorted(files):
            if not filter_func(str(fpath)):
                continue
            hash_object.update(str(fpath).encode("utf-8"))
            fullpath = str(basepath / fpath)
            if fpath.name.lower().startswith("makefile"):
                makefile_content = (basepath / fpath).read_bytes()
                makefile_content = makefile_normalizer(makefile_content)
                hash_object.update(makefile_content)
            else:
                with open(fullpath, "rb") as f:
                    # read file in chunks of 32kb
                    # in order to support large files ( constants.obj )
                    while True:
                        chunk = f.read(1024 * 32)
                        if not chunk:
                            break
                        hash_object.update(chunk)
            if debug:
                hash_log.write(f"\t{str(fpath)} -> {hash_object.hexdigest()}\n")
        if debug:
            hash_log.write(
                f"Final hash of {build_dir} is {hash_object.hexdigest().lower()}\n"
            )
        return hash_object.hexdigest().lower()
    finally:
        if hash_log:
            hash_log.close()


def write_binhash_file(
    build_dir,
    binhash_filename="constants.hash",
    filter_func: Callable[[str], bool] = is_bin_file,
):
    """Hash all binary input files, so we don't have to keep them ( Usecase: constants.obj / constants.bin )

    Args:
        build_dir (str): Path to build directory
        binhash_filename (str, optional): File to be written within build_dir, defaults to "constants.hash".
        filter_func (Callable[[str], bool], optional): Filter function to determine which files to hash. Defaults to is_bin_file.
    """
    binhash = create_dir_hash([binhash_filename], build_dir, filter_func=filter_func)
    with open(os.path.join(build_dir, binhash_filename), "w", encoding="utf-8") as f:
        f.write(binhash)


class BuildCache(ABC):
    """
    Abstract base class for build cache implementations
    """

    @abstractmethod
    def retrieve_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        from_sources_filter_func: Callable[[str], bool] = is_source,
    ) -> Tuple[bool, Optional[str]]:
        """
        Retrieves the build cache artifacts for the given build directory,
        so that ideally no compilation needs to take place.

        Args:
            cmds (_type_): Build commands, these will be part of the hash used to calculate a lookup key
            build_dir (str): Build directory. The source files, Makefile and some other files will be hashed and used to
                             determine the build cache key.
            from_sources_filter_func (Callable[[str], bool], optional): Filter function, which may be used to determine which files are being considered source files. Defaults to is_source.

        Returns:
            Tuple[bool, Optional[str]]: A tuple indicating whether the build cache was successfully retrieved, and a cache key (which should be passed on to store_build_cache on rebuild )
        """
        ...

    @abstractmethod
    def store_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        cache_key: str,
        filter_func: Callable[[str], bool] = is_cache_artifact,
    ) -> bool:
        """
        Store the build cache artifacts

        Args:
            cmds ( List[str]): Build commands, these will be part of the hash used to calculate a lookup key
            build_dir (str): Path to build directory to retrieve build artifacts from
            cache_key (str): Cache key, as returned from retrieve_build_cache
            filter_func (Callable[[str], bool], optional): Filter function, which may be used to determine which files are being considered cacheable artifact files. Defaults to is_cache_artifact.

        Returns:
            bool: Whether the artifacts were successfully stored
        """
        ...

    def maybe_cleanup(
        self, lru_retention_hours: int = 72, cleanup_max_age_seconds: int = 3600
    ):
        """
        Maybe clean up the build cache if its been longer than `cleanup_max_age_seconds` that it has been cleaned up

        Args:
            lru_retention_hours (int, optional): How many hours should unused elements be retained in the cache? Defaults to 72.
            cleanup_max_age_seconds (int, optional): Cleanup interval in seconds. Defaults to 3600.
        """
        pass

    def cleanup(self, retention_hours: int = 72):
        """Do a cache cleanup.

        Args:
            retention_hours (int, optional): How many hours should unused elements be retained in the cache? Defaults to 72.
        """
        pass


class NoBuildCache(BuildCache):
    def __init__(self):
        """
        Dummy build cache implementation which does nothing.

        For method docstrings, see parent class.
        """
        _LOGGER.info("Build cache disabled")

    def retrieve_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        from_sources_filter_func: Callable[[str], bool] = is_source,
    ) -> Tuple[bool, Optional[str]]:
        return False, None

    def store_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        cache_key: str,
        filter_func: Callable[[str], bool] = is_cache_artifact,
    ) -> bool:
        pass


class FileBasedBuildCache(BuildCache):
    def __init__(
        self,
        cache_dir,
        lru_retention_hours=72,
        cleanup_max_age_seconds=3600,
        debug=True,
    ):
        """Filesystem based build cache.

        For method docstrings, see parent class.

        Args:
            cache_dir (str): Path to store cache data below. Should be an empty, temporary directory with enough space to hold the cache contents. Will be written to and deleted in!
            lru_retention_hours (int, optional): Retention time for *unused* cache entries. Defaults to 72.
            cleanup_max_age_seconds (int, optional): Minimum time between cache cleanups in seconds. After this time, a new cleanup gets triggered on next cache retrieval. Defaults to 3600.
            debug (bool, optional): Whether to enable debugging cache key creation ( see debug parameter of create_dir_hash). Defaults to True. May be left at True, as it is usually helpful and  does not hurt performance.
        """
        self.cache_dir = cache_dir
        self.lru_retention_hours = lru_retention_hours
        self.cleanup_max_age_seconds = cleanup_max_age_seconds
        self.debug = debug
        _LOGGER.info(
            f"Using file-based build cache, cache directory = {self.cache_dir}"
        )

    def retrieve_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        from_sources_filter_func: Callable[[str], bool] = is_source,
    ) -> Tuple[bool, Optional[str]]:
        """See docstring of implemented method interface in parent class"""
        if should_skip_build_cache():
            _LOGGER.info(f"CACHE: Skipped build cache for {build_dir}")
            return False, None
        self.maybe_cleanup(self.lru_retention_hours, self.cleanup_max_age_seconds)
        cache_dir = self.cache_dir
        dir_hash = create_dir_hash(
            cmds, build_dir, filter_func=from_sources_filter_func, debug=self.debug
        )
        key_cache_dir = os.path.join(cache_dir, dir_hash)
        if os.path.exists(key_cache_dir):
            _LOGGER.info(f"CACHE: Using cached build results for {build_dir}")
            target_basepath = Path(build_dir)
            src_basepath = Path(key_cache_dir)
            copy_files = [
                p.relative_to(src_basepath)
                for p in src_basepath.rglob("*")
                if not p.is_dir()
            ]
            for filepath in copy_files:
                target_path = target_basepath / filepath
                target_parent = target_path.parent
                src_path = src_basepath / filepath
                if target_parent != target_basepath:
                    os.makedirs(str(target_parent), exist_ok=True)
                shutil.copy(
                    str(src_path),
                    str(target_path),
                    follow_symlinks=True,
                )  # Using shutil.copy intentionally instead of copy2, so the file modification time is updated, and file owner
                # is not copied. When you retrieve the file from cache, it is yours.
                _LOGGER.debug(f"CACHE: retrieved {filepath}")
            # make sure the last modified timestamp is updated, so we can
            # evict cache directories which are too old using a separate script
            os.utime(key_cache_dir)
            return True, dir_hash
        _LOGGER.info(f"CACHE: No results found for {build_dir}")
        return False, dir_hash

    def store_build_cache(
        self,
        cmds: List[str],
        build_dir: str,
        cache_key: str,
        filter_func: Callable[[str], bool] = is_cache_artifact,
    ) -> bool:
        """See docstring of implemented method interface in parent class"""
        cache_dir = self.cache_dir
        key_cache_dir = os.path.join(cache_dir, cache_key)

        # We create a temporary directory first, so we can do an
        # atomic update later to prevent race conditions
        # in a distributed / parallel build setting
        random_str = secrets.token_hex(16)

        # the temp_cache_dir will be renamed to key_cache_dir
        # atomically later. It needs to be on same file system
        # for atomic rename, so we put it into the same folder.
        temp_cache_dir = key_cache_dir + f".{random_str}.tmp"
        try:
            os.makedirs(temp_cache_dir, exist_ok=False)
        except OSError:
            _LOGGER.warn(
                f"CACHE: Failed to create tempdir {temp_cache_dir}. Cannot write cache entries."
            )
            return False
        basepath = Path(build_dir)
        target_basepath = Path(temp_cache_dir)
        copy_files = [
            p.relative_to(basepath) for p in basepath.rglob("*") if not p.is_dir()
        ]
        for filepath in copy_files:
            src_path = basepath / filepath
            if not filter_func(str(filepath)):
                continue

            target_path = target_basepath / filepath
            target_parent = target_path.parent
            if target_parent != target_basepath:
                os.makedirs(str(target_parent), exist_ok=True)
            shutil.copy2(
                str(src_path),
                str(target_path),
                follow_symlinks=True,
            )  # Use copy2, so the file metadata (incl. last modified time) is preserved
            _LOGGER.info(f"CACHE: storing {filepath} into {key_cache_dir}: ")
        try:
            os.rename(
                temp_cache_dir, key_cache_dir
            )  # Atomic update to prevent race condition
            return True
        except OSError:
            _LOGGER.info(
                f"CACHE: update race conflict - {key_cache_dir} already exists. (Note: No error! This can be expected to happen occasionally.))"
            )
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
            return False

    def maybe_cleanup(
        self, lru_retention_hours: int = 72, cleanup_max_age_seconds: int = 3600
    ):
        """See docstring of implemented method interface in parent class"""
        last_cleaned_seconds = file_age(os.path.join(self.cache_dir, ".last_cleaned"))
        if last_cleaned_seconds > cleanup_max_age_seconds:
            self.cleanup(lru_retention_hours)

    def cleanup(self, lru_retention_hours: int = 72):
        """See docstring of implemented method interface in parent class"""
        _LOGGER.info(
            f"CACHE: Cleaning up build cache below {self.cache_dir}. Folders last used more than {lru_retention_hours} hours ago will be deleted."
        )
        touch(os.path.join(self.cache_dir, ".last_cleaned"))
        if os.path.isdir(self.cache_dir):
            now = datetime.now()
            age_limit = timedelta(hours=lru_retention_hours)

            for dirpath in os.scandir(self.cache_dir):
                if os.path.isdir(dirpath):
                    # Get the modification time of the directory and convert it to a datetime object
                    mtime = os.path.getmtime(dirpath)
                    modification_time = datetime.fromtimestamp(mtime)

                    # Check if the directory is older than N hours
                    if now - modification_time > age_limit:
                        _LOGGER.info(f"CACHE: Deleting {dirpath}")
                        shutil.rmtree(dirpath)
