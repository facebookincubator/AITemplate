import os.path as path


def save_profile_cache(remote_cache_file_path, cache_path):
    with open(cache_path, "rb") as f:
        with open(remote_cache_file_path, "wb") as target:
            target.write(f.read())


def load_profile_cache(remote_cache_file_path, cache_bytes):
    if path.isfile(remote_cache_file_path):
        with open(remote_cache_file_path, "rb") as cache_content:
            cache_bytes.write(cache_content.read())
