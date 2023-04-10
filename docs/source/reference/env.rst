Environment Variables
=====================
AITemplate uses environment variables to control the behavior of codegen and profiling.
The environment variables used in AITemplate are listed here.

Codegen
-------

**NUM_BUILDERS**: The number of CPU jobs running in parallel during codegen. It controls both the profiler codegen and the final .so codegen. It's set to 12 in NIGHTLY jobs. Internally, it's set to 12 for normal tests and 24 for heavy tests. By default, the builder uses all the available CPUs for building.

**AIT_RECOMPILE**: If set to "0", it skips compilation for the .so and reuses the previously compiled ones. It is used to speed up local testing. The default value is "1" to always recompile.

**AIT_NDEBUG**: If set to "1", compile with `NDEBUG`, disabling debug assertions. Recommended for production builds. "1" by default.

**AIT_COMPILER_OPT**: The optimization level for a compiler, which is directly passed to the host compiler command line. AITemplate host code may be very light in certain cases, so there is nothing to optimize for a host compiler. Thus, there is no need to make host compiler perform time costly optimizations. It may be very useful to use "-O0" value for debugging GPU kernels. "-O3" by default.

**AIT_TIME_COMPILATION**: If set to "1", time each make command at the compilation time. This helps us to do compilation time analysis. Requires to install `time <https://man7.org/linux/man-pages/man1/time.1.html>`_ package.

Profiling
---------

**CACHE_DIR**: The directory for the profiling cache. If unset, it defaults to `~/.aitemplate`.

**FLUSH_PROFILE_CACHE**: If set to "1", it removes the cache file and recreates an empty one.

**DISABLE_PROFILER_CODEGEN**: Normally in CI we randomly choose two profilers to codegen. If set to "1", this flag disables profiler codegen completely to speed up long running tests so that the tests don't time out. The default value is "0".

**CUDA_VISIBLE_DEVICES**: This one is from CUDA itself. It's used to set the number of GPU devices available for profiling. Set to "0,1,2,3,4,5,6,7" to speed up profiling. For benchmarking, it's useful to set to a particular device to lower noise.

**HIP_VISIBLE_DEVICES**: This one is from ROCm itself. It's used to set the number of GPU devices available for profiling. Set to "0,1,2,3,4,5,6,7" to speed up profiling. For benchmarking, it's useful to set to a particular device to lower noise.

**FORCE_PROFILE**: If set to "1", it will do profiling regardless in_ci_env and disable_profiler_codegen. For non-NIGHTLY CI, we do not do profiling, and we could use FORCE_PROFILE=1 in these CI to do runs with codegen, compile, and profile.

**COMBINE_PROFILER_MULTI_SOURCES**: Whether to combine multiple profiler sources per target. "0" - Disabled, "1" - Enabled (default).

**FORCE_ONE_PROFILER_SOURCE_PER_TARGET**: Whether to combine multiple profiler sources per target into one. "0" - Disabled (default), "1" - Enabled.

OSS CI
------

**CI_FLAG**: It is set to "CIRCLECI" in OSS CI to indicate we're in OSS CI environment. The behavior of the profiler and codegen is different in CI to speed up testing. Profiling itself for gemm/conv ops is disabled in CI. But we still compile two random profilers to make sure the profiler codegen is not broken.

**AIT_BUILD_DOCS**: If set to "1", it will create a fake CUDA target to enable doc building in Github Actions.

Miscellaneous
-------------

**LOGLEVEL**: It is used to control the logging level in Python. The default value is "INFO". "DEBUG" is useful for debugging.

**AIT_PLOT_SHORTEN_TENSOR_NAMES**: If set to "1", shorten too long tensor names for a plot of a model graph, thus making a plot much easier to analyze visually. "0" by default.
