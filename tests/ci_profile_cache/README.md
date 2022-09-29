# Profiling Database for CI (Deprecated)

Profile Cache DB for CI is deprecated. Now CI will select the algorithm with the smallest tiling size and smallest alignments for CI.

The selection function is defined at: `backend/target.py:  Target:select_minimal_algo` and specialized in each backend target implementation.
