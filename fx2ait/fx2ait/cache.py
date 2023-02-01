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
import os.path as path


def save_profile_cache(remote_cache_file_path, cache_path):
    with open(cache_path, "rb") as f:
        with open(remote_cache_file_path, "wb") as target:
            target.write(f.read())


def load_profile_cache(remote_cache_file_path, cache_bytes):
    if path.isfile(remote_cache_file_path):
        with open(remote_cache_file_path, "rb") as cache_content:
            cache_bytes.write(cache_content.read())
