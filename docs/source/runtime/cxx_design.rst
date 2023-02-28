================
C++ Runtime Note
================

`Model` v.s. `ModelContainer`
=============================

These are the two main classes involved in the C++ runtime implementation:

* The bulk of the runtime implementation is in the `Model` class.
* The `ModelContainer` class stores a set of shared constants and a collection of `Model` instances. Almost all functions in `model_interface.h` forward to a method in `ModelContainer`. When `Run` is invoked, `ModelContainer` looks for an available `Model`, or blocks until one becomes available (see the section on asynchronous predictions). It then forwards the run request to the runtime.

Code Structure
==============

Some important files:

1. `include/model_interface.h`: The interface that we expose in the compiled `.so`.
2. `include/model_container.h`: The bulk of the `ModelContainer` implementation.

Some files are generated at compile time. These include:

* `model-generated.h`: The implementation of the `Model`.
* `model_container_base.cu`: A small part of the implementation for `ModelContainer` that needs to be generated. `ModelContainer` inherits from `ModelContainerBase`, and `ModelContainerBase`'s implementation lives in this file. See `model_container.h` for more details.

All codegen templates can be found in `backend/main_templates.py`.
The codegen implementation is in `backend/codegen.py`.

Note that many of the headers in this directory rely on generated code and thus cannot be `#include` -d in external projects.
`model_interface.h` is an exception.
