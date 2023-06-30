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
from hashlib import sha1

from aitemplate import backend
from aitemplate.backend import registry


_LOGGER = logging.getLogger(__name__)


def get_profiler_filename(func_attrs, op_class):
    """
    Generate a filename for a profiler that benchmarks multiple instances.
    """
    target = backend.target.Target.current()

    op_type = func_attrs["op"]
    all_op_names = list(func_attrs["op_instance"].keys())
    encoded_str = sha1((";".join(all_op_names)).encode("utf-8")).hexdigest()

    if target.use_dummy_profiling_results():
        # we don't use cache
        return f"{op_type}_{encoded_str}"
    else:
        cache_ver = target.get_profile_cache_version(op_class)
        return f"{op_type}_{encoded_str}_{cache_ver}"


def filter_op_instances(func_attrs, x_shapes):
    """
    Filter out some of the func's op instances using the filter function.
    """
    target = backend.target.Target.current()
    func_key = "{target}.{op}.filter".format(
        target=target.name(),
        op=func_attrs["op"],
    )
    filter_func = registry.get(func_key)

    op_names_to_keep = set()
    for x_shape in x_shapes:
        for op_name in func_attrs["op_instance"]:
            if filter_func(op_name, func_attrs, x_shape):
                op_names_to_keep.add(op_name)

    return {
        op_name: op
        for op_name, op in func_attrs["op_instance"].items()
        if op_name in op_names_to_keep
    }


def generate_profiler_sources(func_attrs, op_class, workdir, shape_template):
    """
    Generate profiler sources for the func.
    """
    target = backend.target.Target.current()
    func_key = "{target}.{op}.gen_profiler".format(
        target=target.name(),
        op=func_attrs["op"],
    )
    gen_profiler_func = registry.get(func_key)

    if target.name() == "rocm":
        return gen_profiler_func(func_attrs, workdir, shape_template)

    profiler_filename = get_profiler_filename(func_attrs, op_class)
    _LOGGER.info(f"generating {profiler_filename=}")
    return gen_profiler_func(
        func_attrs,
        workdir,
        profiler_filename,
        shape_template,
    )
