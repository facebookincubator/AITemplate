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

import fcntl
import os
import time
from typing import Callable, Optional


class FileLock:
    def __init__(
        self,
        path,
        timeout=3600.0,
        retry_interval=0.2,
        lock_contention_callback: Optional[Callable] = None,
    ):
        """
        File locking context manager. Acts like an inter-process mutex
        on a given file. The lock will be released when the context manager
        exits. It has a timeout parameter, which triggers a TimeoutError
        on expiry in order to prevent infinite loops or deadlocks.

        Implementation note:
            Attempts to open the given file for writing, and
            acquires an exclusive file lock on success. If the file cannot be opened
            or the lock cannot be acquired, it retries after retry_interval
            seconds until success or a timeout occurs.

            Uses fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            to acquire the file lock.

        Usage:

            with FileLock('/path/to/file.lock', timeout=10):
                ...

        Args:
            path (str ): Path to lockfile. Will be created as an empty file
            if it does not exist. Will not be deleted on exit.
            timeout (int, optional): Timeout in seconds. Defaults to 3600.
            retry_interval (float, optional): Retry interval in seconds. Defaults to 0.2.
            lock_contention_callback(Callable,optional): A callback that will be invoked without arguments
                                      in the case that a lock cannot be acquired on first attempt.
                                      May be used for logging purposes.
        """
        self.path = path
        self.timeout = timeout
        self.lock_file = None
        self.retry_interval = retry_interval
        self.lock_contention_callback = lock_contention_callback

    def __enter__(self):
        start_time = time.monotonic()
        attempts = 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                self.lock_file = open(self.path, "w")
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if self.lock_file is not None:
                    self.lock_file.close()
                    self.lock_file = None
                if attempts == 0 and self.lock_contention_callback is not None:
                    lock_contention_callback = self.lock_contention_callback
                    lock_contention_callback()
                attempts += 1
                if (
                    self.timeout is not None
                    and time.monotonic() - start_time >= self.timeout
                ):
                    raise TimeoutError(
                        f"Timeout waiting for lock on {self.path}"
                    ) from None
                time.sleep(self.retry_interval)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        finally:
            self.lock_file.close()
            self.lock_file = None
        return False
