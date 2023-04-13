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
Util functions to handle file or network io
"""
import hashlib
import logging
import os
import tarfile
import time
from io import BytesIO, FileIO
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

_LOGGER = logging.getLogger(__name__)


def touch(file_path):
    """
    Emulates the Linux 'touch' command by creating an empty file if it doesn't exist, or updating the modified timestamp if it does.

    :param file_path: str: The path to the file to be created or updated.
    :return: None
    """
    if not os.path.exists(file_path):
        p = Path(file_path)
        # ensure parent directory exists
        os.makedirs(str(p.parent), exist_ok=True)
        open(file_path, "w").close()

    # Update the modified timestamp
    os.utime(file_path)


def file_age(file_path):
    """
    Returns the age of a file in seconds since its last modified timestamp.

    :param file_path: str: The path to the file.
    :return: float: The age of the file in seconds.
    """
    if not os.path.isfile(file_path):
        return 3600 * 24 * 1000.0

    # Get the current time and the file's last modification time
    current_time = time.time()
    file_mtime = os.path.getmtime(file_path)

    # Calculate the file age in seconds
    file_age_seconds = current_time - file_mtime

    return file_age_seconds


def file_sizes(directory, filter_function=None):
    total_size = 0
    for root, _dirs, files in os.walk(directory):
        for _file in files:
            file_path = os.path.join(root, _file)
            if filter_function is not None and filter_function(file_path):
                total_size += os.path.getsize(file_path)

    return total_size


# Utility functions to be used by (not yet existing) distributed cache implementations
# to minimize the amount of network roundtrips and network bandwidth needed


def create_archive(
    directory_path: str,
    filter_func: Callable[[str], bool] = None,
    output_file: Optional[str] = None,
) -> Optional[bytes]:
    """Create tar.gz archive in-memory and return the archive contents as
    a bytes object.

    Args:
        directory_path (str): Directory to create archive of.
        filter_func (_type_, optional): A function which, being passed a filename,
                                        returns whether to include it or not.
                                        Defaults to None (include all).
        output_file (str): Output filename to write the archive to. Usually it ends on .tar.gz.
                           If set to None ( default), the archive will not be written to
                           file but returned as a bytes object.

    Returns:
        Optional[bytes]: Archive contents as a bytes object if output_file was not None
    """
    # Archive files in a directory.

    # Create an in-memory bytes buffer
    if output_file is None:
        buffer = BytesIO()
    else:
        buffer = FileIO(output_file, mode="w+")

    # Determine the appropriate compression mode
    compression_mode = None
    compression_mode = "w:gz"

    # Create a new archive file
    with tarfile.open(fileobj=buffer, mode=compression_mode) as archive:
        # Walk through the directory tree and add each file to the archive
        for root, _, files in os.walk(directory_path):
            for _file in files:
                # Check if the file should be included based on the filter function
                if filter_func is not None:
                    if not filter_func(_file):
                        continue

                # Calculate the relative path of the file
                relative_path = os.path.relpath(
                    os.path.join(root, _file), directory_path
                )

                # Add the file to the archive with the relative path
                archive.add(os.path.join(root, _file), arcname=relative_path)

    # Get the bytes from the buffer
    if output_file is not None:
        buffer.close()
        return None
    buffer.seek(0)
    compressed_bytes = buffer.read()

    return compressed_bytes


def extract_archive(
    archive_data: BinaryIO, target_directory: str, overwrite: bool = False
):
    """Extract a tar.gz archive (written for example via create_archive) from a bytes buffer
    into a target directory.

    Args:
        archive_data (BinaryIO): BinaryIO object ( typicall BytesIO or FileIO ) of the tar.gz archive to be extracted.
        target_directory (str): Target directory to extract to.
        overwrite (bool, optional): Whether to overwrite files or not.
                                    If False, files will be silently skipped
                                    if they already exist. Defaults to False.
    """
    archive = tarfile.open(fileobj=archive_data, mode="r:gz")

    # Extract the archive contents into the target directory
    for member in archive.getmembers():
        # Calculate the full path of the extracted file or directory
        target_path = os.path.join(target_directory, member.name)

        # Check if the file or directory already exists
        if os.path.exists(target_path):
            if not overwrite:
                _LOGGER.debug(
                    f"extract_archive: Skipping extraction of file to {os.path.abspath(target_path)}: A file at that path already exists, and overwrite is not enabled."
                )
                continue
            else:
                _LOGGER.debug(
                    f"extract_archive: Replacing existing file at {os.path.abspath(target_path)} with file from archive."
                )
                os.remove(target_path)

        # Extract the file or directory from the archive
        archive.extract(member, target_directory)

    # Close the archive object
    archive.close()


def copytree_with_hash(
    src_path: Union[Path, str],
    dst_path: Union[Path, str],
    buffer_size=1024 * 1024,
    hash: Optional[hashlib.sha256] = None,
    max_depth: int = 20,
) -> Optional[str]:
    """Copy a directory and its contents recursively, while at the same time calculating a hash over each file and filename.

    :param src_path: Path: The path to the source directory.
    :param dst_path: Path: The path to the destination directory.
    :param buffer_size: int: The buffer size to read and write data in.
    :param hash: Optional[hashlib.sha256]: The hash to use for calculating the hash. ( Default: None)
    :max_depth: int : The maximum recursion depth. Default: 20
    :return: None, if a hash instance was passed. Otherwise, the hash of the copied data and path names.
    """

    if hash is None:
        hash_obj = hashlib.sha256()
    else:
        hash_obj = hash
    if isinstance(src_path, str):
        src_path = Path(src_path)
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    if dst_path.exists():
        dst_path = dst_path.resolve()
        if not dst_path.is_dir():
            raise OSError("Target path exists and is not a directory.")
        dst_path = dst_path / src_path.name
    if src_path.is_file():
        hash_obj.update(dst_path.name.encode("utf-8"))
        # Copy the file to the destination
        with open(dst_path, "wb") as dst_file:
            with open(src_path, "rb") as src_file:
                while True:
                    data = src_file.read(buffer_size)
                    if not data:
                        break
                    hash_obj.update(data)
                    dst_file.write(data)
    elif src_path.is_symlink():
        new_src_path = src_path.resolve()
        copytree_with_hash(new_src_path, dst_path, buffer_size, hash_obj, max_depth - 1)
    elif src_path.is_dir():
        # Recursively copy the directory contents
        os.makedirs(dst_path, exist_ok=True)
        for sub_path in sorted(src_path.iterdir()):
            sub_dst_path = dst_path / sub_path.name
            copytree_with_hash(
                sub_path, sub_dst_path, buffer_size, hash_obj, max_depth - 1
            )
    else:
        raise OSError(f"Source path {src_path} is neither file, directory nor symlink.")
    if hash is None:
        return hash_obj.hexdigest()
