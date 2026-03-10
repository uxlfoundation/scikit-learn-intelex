.. Copyright contributors to the oneDAL project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. include:: substitutions.rst

====================
Building from Source
====================

Components
----------

The |sklearnex| predominantly functions as a frontend to the |onedal| by leveraging it as a backend for |sklearn| calls. In order to build the |sklearnex|, it's necessary to have a version of the |onedal| as a shared library already built somewhere along with its headers - for example, by using the Python packages ``dal`` + ``dal-devel`` (conda) / ``daal`` + ``daal-devel`` (PyPI), or the system-wide `offline installer <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal-download.html>`__, or by `building oneDAL from source <https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md>`__.

.. note:: Python packages ``dal`` (conda) and ``daal`` (PyPI) provide the same components, but due to naming availability in these repositories, they are distributed under different names.

As a library, the |sklearnex| consists of a Python codebase with Python extension modules written in C++ and Cython, with some of those modules being optional. These extension modules require compilation before being used, for which a C++ compiler along with other dependencies is required. In the case of GPU-related modules, a SYCL compiler (such as `Intel's DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`__) is required, and in the case of distributed mode, whether on CPU or on GPU, an MPI backend is required, such as `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`__.

The extension modules are as follows:

- ``daal4py``: the source code for this module is auto-generated from the headers of the |onedal| as a Cython file through the code under the folder `generator <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/generator>`__, along with other C++ source files. This module is mandatory. It provides the necessary bindings for the DAAL interface - see :doc:`about_daal4py` for details. It will contain also the necessary MPI bindings for distributed computations on CPU if building with distributed mode (see :doc:`distributed_daal4py` for details), and the necessary bindings for streaming mode if that functionality is built.
- ``_onedal_py_host``: this module provides PyBind11-generated bindings over the oneAPI interface of the |onedal| for CPU (host). This module is mandatory.
- ``_onedal_py_dpc``: this module provides PyBind11-generated bindings over the oneAPI interface of the |onedal| for GPU (DPC++). This module is optional, and requires a SYCL compiler. If the oneDAL backend is compiled from source, it must also have been built with its DPC++ component in order to build this module. See :doc:`oneapi-gpu` for more information.
- ``_onedal_py_spmd`` (Linux*-only): this module provides PyBind11-generated bindings over SPMD implementations (distributed mode on GPU) using the oneAPI interface of the |onedal| - see :doc:`distributed-mode` for details. This module is optional, and requires both a SYCL compiler and an MPI backend, along with its headers. It requires the ``_onedal_py_dpc`` module to also be built.

**Note that all of the optional components are built by default** (see rest of this page for how to enable or disable specific components).

Build Requirements
------------------

The |sklearnex| `dependencies-dev <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/dependencies-dev>`__ provides versioned mandatory dependencies for building from source and usage in CI jobs. Note however that this file does not contain all of the necessary dependencies for distributed mode, nor does it contain compiler-related dependencies, and it is not strongly necessary to install the exact same versions as in that file for local development purposes.

Python dependencies
~~~~~~~~~~~~~~~~~~~

To install the necessary Python dependencies:

- Using ``conda``:

.. code-block:: bash

    conda install -c conda-forge numpy cython jinja2 pybind11 "setuptools<=79"

- Using ``pip``:

.. code-block:: bash

    pip install numpy cython jinja2 pybind11 "setuptools<=79"

.. hint:: Using the compiled library after building it has a different set of requirements, such as the |sklearn| package along with its dependencies. Executing the tests also adds additional dependencies such as ``pytest``. These test dependencies can be installed from file `requirements-test.txt <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/requirements-test.txt>`__.

Non-Python dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Apart from Python libraries and from the |onedal| (version ``2021.4`` or higher), the following dependencies are needed in order to compile the |sklearnex|:

- A C++ compiler.
- clang-format.
- CMake.
- A DPC++ compiler (required for GPU components).
- An MPI backend and its headers (required for distributed components).

The easiest way to install the necessary dependencies that are not Python libraries is with conda.

- On Linux*:

.. code-block:: bash

    conda install -c conda-forge \
        cmake clang-format cxx-compiler `# mandatory dependencies` \
        dpcpp-cpp-rt dpcpp_linux-64 `# required for GPU mode` \
        impi-devel impi_rt `# required for distributed mode`

- On Windows*:

.. code-block:: bash

    conda install -c conda-forge ^
        cmake clang-format cxx-compiler ^
        dpcpp-cpp-rt dpcpp_win-64 ^
        impi-devel impi_rt

Some of these dependencies can also be installed from PyPI:

.. code-block:: bash

    pip install clang-format impi-devel impi_rt

Note however that, if installing Intel's MPI from PyPI instead of from conda, it will be necessary to manually set the environment variable ``$MPIROOT``, while the conda distribution of Intel's MPI comes with an activation script that sets up this variable.

Instructions
------------

Setting environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before compiling the |sklearnex|, it's necessary to set up some environment variables to point to the installation paths of dependencies.

OneDAL
******

An environment variable ``$DALROOT`` must be set to the path containing the |onedal| library, such that the shared objects (``.so`` / ``.dll``) will be findable under the path ``$DALROOT/lib``. This environment variable can be set in different ways:

    - If using an offline installer for the |onedal|, this variable will be set automatically when sourcing the general activation script for oneAPI products, which can be done as follows, assuming a Linux* system:

        .. code-block:: bash

            source /opt/intel/oneapi/setvars.sh

    - If building the |onedal| from source, it will be set automatically when sourcing the generated environment activation script - see the `instructions on the oneDAL repository <https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md#installation-steps>`__ for more details.

    - Otherwise, the variable can be set manually. For example, if installing oneDAL through ``conda``, assuming a Linux* system:

        .. code-block:: bash

            export DALROOT="$CONDA_PREFIX"

.. important:: If the |onedal| is not under a default system path, in order to be able to load it after compiling the |sklearnex|, its path must be added to an environment variable such as ``$LD_LIBRARY_PATH``, or the |sklearnex| must be built with argument ``--abs-rpath`` (see rest of this document for details).

MPI
***

If building with distributed mode, an environment variable ``$MPIROOT`` must be set to the path containing the MPI library, such that the shared objects (such as ``libmpi.so``) will be findable under ``$MPIROOT/lib`` and the headers under ``$MPIROOT/include``. Alternatively, environment variable ``$I_MPI_ROOT``, which is used by Intel's MPI, will be used if it is defined while ``$MPIROOT`` isn't. If using Intel's MPI, this variable can be set in different ways:

- If installing IMPI (Intel's MPI) from conda, the variable will be set automatically upon activation of the conda environment.
- If using an offline installer for IMPI, this variable will be set automatically when sourcing the general activation script for oneAPI products, which can be done as follows, assuming a Linux* system:

    .. code-block:: bash

        source /opt/intel/oneapi/setvars.sh

- Otherwise, the variable can be set manually. For example, if installing some MPI other than IMPI through ``conda``, assuming a Linux* system:

    .. code-block:: bash

        export MPIROOT="$CONDA_PREFIX"

.. _build_script:

Build using ``setup.py``
~~~~~~~~~~~~~~~~~~~~~~~~

With all of the necessary requirements and environment variables already set up, the library can be installed from source as follows:

.. code-block:: bash

    python setup.py install

.. hint:: See the rest of this document for build-time options, such as disabling distributed mode or disabling GPU mode.

To install it in development mode:

.. code-block:: bash

    python setup.py develop

To build the extensions in-place without installing (recommended for local development):

.. code-block:: bash

    python setup.py build_ext --inplace --force # builds daal4py
    python setup.py build # builds onedal extension modules

.. hint:: If building the library in-place without installing, it's then necessary to set environment variable ``$PYTHONPATH`` to point to the root of the repository in order to be able to import the modules in Python.

Build using conda
~~~~~~~~~~~~~~~~~

The |sklearnex| can also be easily built from source with a single command using ``conda-build``. 

Requirements
************

The following are required in order to use ``conda-build``:

- Any ``conda`` distribution (`Miniforge <https://github.com/conda-forge/miniforge>`__ is recommended).
- ``conda-build`` package installed in a conda environment:

    .. code-block:: bash

        conda install -c conda-forge conda-build

- On Windows*, an **external** installation of the MSVC compiler **version 2022** is required by default. Other versions can be specified in `conda-recipe/conda_build_config.yaml <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/conda-recipe/conda_build_config.yaml>`__ if needed.
- Optionally, for DPC++ (GPU) support on Windows*, environment variable ``%DPCPPROOT%`` must be set to point to the DPC++ compiler path.

Instructions
************

When building with conda, if the environment variables for the |onedal| and MPI are not set, those dependencies will be managed by conda instead, which will use their respective conda packages and set the environment variables internally during the builds.

.. hint::

    If there was any previous from-source installation or build from a different environment, one might need to delete the ``build/`` folder and the generated ``.so`` / ``.pyd`` modules:

    .. code-block:: bash

        rm -Rf build daal4py/*.so onedal/*.so

To create and verify the conda package for this library, execute the following command from the root of the repository **after installing conda-build**:

.. code-block:: bash

    conda build .

.. hint::

    To clear build environments afterwards, one can issue the following command from the conda environment that executed ``conda build``:

    .. code-block:: bash

        conda build purge

Build-time Options
------------------

The setup script accepts many configurable options, some controllable through environment variables and others controllable through command line arguments. For example:

.. code-block:: bash

    NO_DIST=1 python setup.py build_ext --inplace --force --abs-rpath

Additionally, the tools used by the build backend can also be passed custom configurations through environment variables such as ``$CXX``, ``$CXXFLAGS``, ``$LDFLAGS``, etc. For example:

.. code-block:: bash

    NO_DIST=1 LDFLAGS="-fuse-ld=lld" python setup.py build --using-lld

Environment variables
~~~~~~~~~~~~~~~~~~~~~

The following environment variables can be used to control setup aspects:

- ``SKLEARNEX_VERSION``: sets the package version.
- ``DALROOT``: sets the |onedal| path.
- ``MPIROOT``: sets the path to the MPI library. If this variable is not set but ``I_MPI_ROOT`` is found, will use ``I_MPI_ROOT`` instead. Not used when using ``NO_DIST=1``.
- ``NO_DIST``: set to '1', 'yes' or alike to build without support for distributed mode.
- ``NO_STREAM``: set to '1', 'yes' or alike to build without support for streaming mode.
- ``NO_DPC``: set to '1', 'yes' or alike to build without support of oneDAL DPC++ interfaces.
- ``MAKEFLAGS``: the last `-j` flag determines the number of threads for building the onedal extension. It will default to the number of CPU threads when not set.

.. note:: The ``-j`` flag in the ``MAKEFLAGS`` environment variable is superseded in ``setup.py`` modes which support the ``--parallel`` and ``-j`` command line flags.

Command line arguments
~~~~~~~~~~~~~~~~~~~~~~

The following additional arguments are accepted in calls to the ``setup.py`` script:

- ``--abs-rpath`` (Linux*-only): will make it add the absolute path to the |onedal| shared objects (``.so`` files) to the rpath of the |sklearnex| shared object files in order to load them automatically. This is not necessary when installing through ``pip`` or ``conda``, but can be helpful for development purposes when using a from-source build of the |onedal| that resides in a custom folder, as it won't assume that its files will be found under default system paths.
- ``--debug``: builds modules with debugging symbols and assertions enabled. Note that on Windows*, this will only add debugging symbols for the ``_onedal_py`` extension modules, but not for the ``daal4py`` extension module.
- ``--using-lld`` (Linux*-only): makes the setup script avoid passing arguments that are not supported by LLVM's LLD linker, such as strong stack protection. This flag is required when building with the LLD linker (which can be achieved by setting environment variable ``$LDFLAGS="-fuse-ld=lld"``), but note that it **does not make the build script use LLD**, only avoids adding arguments that it doesn't support.

Apart from these, standard arguments recognized by the build libraries can also be passed in the same call - for example, to install without checking for dependencies:

.. code-block:: bash

    python setup.py install --single-version-externally-managed --record=record.txt
    python setup.py develop --no-deps


Tips
----

Incremental Compilation
~~~~~~~~~~~~~~~~~~~~~~~

The compiled modules are a mixture of Cython and PyBind11. Compilation of the PyBind11 modules is managed through CMake, which offers incremental compilation and parallel compilation, but compilation of the Cython module ``daal4py`` is managed through ``setuptools``, which lacks this feature, and in addition, is compiled under a single thread as it consists of a single large file. Thus, by default, a call to ``python setup.py build`` can take a long time to finish, with most of that time spent in the single-threaded ``daal4py`` compilation.

For local development, in order to speed up setup, one can instead use ``ccache`` in order to avoid recompiling ``daal4py`` modules throughout multiple calls to ``setup.py``. While the build script doesn't have any explicit option for ``ccache``, it can be configured to use it by setting the compiler to something that would execute under it. Example:

.. code-block:: bash

    CC="ccache icx" CXX="ccache icpx" python setup.py build_ext --inplace --force
    CC="ccache icx" CXX="ccache icpx" python setup.py build

Omitting components
~~~~~~~~~~~~~~~~~~~~

When it comes to local development, in many cases the features being developed do not involve an SPMD or GPU component. In such cases, it's faster to compile without those options, and it's likewise usually faster to use the LLD linker and lower the optimization level for the library:

.. code-block:: bash

    NO_DPC=1 NO_DIST=1 CC="ccache icx -O0" CXX="ccache icpx -O0" LDFLAGS="-fuse-ld=lld" \
        python setup.py build_ext --inplace --force --abs-rpath --using-lld
    NO_DPC=1 NO_DIST=1 CC="ccache icx -O0" CXX="ccache icpx -O0" LDFLAGS="-fuse-ld=lld" \
        python setup.py build --abs-rpath --using-lld

Cleaning the build folder
~~~~~~~~~~~~~~~~~~~~~~~~~

When building from source, temporary artifacts are created under a ``/build`` folder. Since some modules use CMake, which is designed for incremental compilation, it will leave pre-compiled objects that it will try to reuse if further builds are executed without modifying the same input files.

However, note that CMake's logic does not consider compatibility of these leftover objects, so for example, if one first compiles the library with a given Python version, and then tries to compile it from the same folder using a different Python version, the leftover artifacts will be incompatible, but CMake will still try to reuse them and fail in the process, with a non-informative error message. Same issue might happen for example if some modules are enabled or disabled across different calls to the ``setup.py`` script.

If experiencing issues during compilation, try removing the existing ``/build`` folder to see if it solve the issues:

.. code-block:: bash

    rm -Rf build

OneTBB runtimes
~~~~~~~~~~~~~~~

When building with the ``--abs-rpath`` option, it will use the |onedal| library version with which it was compiled. |onedal| has dependencies on other libraries such as `oneTBB <https://github.com/uxlfoundation/oneTBB>`__, which is also distributed as a python package through ``pip`` and as a ``conda`` package.

By default, a conda environment will first try to load oneTBB from its own packages if it is installed in the environment, which might cause issues if the |onedal| was compiled with a system oneTBB instead of a conda one.

In such cases, it is advised to either uninstall oneTBB from ``pip``/``conda`` (it will be loaded from the |onedal| library which links to it), or modify the order of search paths in environment variables like ``$LD_LIBRARY_PATH`` to prefer the one with which the |onedal| was compiled instead of the one from ``conda``.

Building with sanitizers
------------------------

Building with ASan
~~~~~~~~~~~~~~~~~~

In order to use AddressSanitizer (ASan) together with the |sklearnex|, it's necessary to:

- Build both the |onedal| and the |sklearnex| with ASan and with debugging symbols (otherwise error traces will not be very informative).
- Preload the ASan runtime when executing the Python process that imports ``sklearnex`` or ``daal4py``.
- Optionally, configure Python to use ``malloc`` as default allocator to reduce the number of false-positive leak reports.

See the `instructions on the oneDAL repository <https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md>`__ for building the library from source with ASAN enabled.

When building this library, the system's default compiler is used unless specified otherwise through variables such as ``$CXX``. In order to avoid issues with incompatible runtimes of ASan, one might want to change the compiler to ICX if the |onedal| was built with ICX (the default for it).

The compiler and flags to build with both ASan and debug symbols can be controlled through environment variables - **assuming a Linux\* system** (ASan on Windows* has not been tested):

.. code-block:: bash

    export CC="icx -fsanitize=address -g"
    export CXX="icpx -fsanitize=address -g"

.. hint:: The Cython module ``daal4py`` that gets built through ``build_ext`` does not do incremental compilation, so one might want to add ``ccache`` into the compiler call for development purposes - e.g. ``CXX="ccache icx -fsanitize=address -g"``.

The ASan runtime used by ICX is the same as the one by Clang. It's possible to preload the ASan runtime for GNU if that's the system's default through e.g. ``$LD_PRELOAD=libasan.so`` or similar. However, one might need to specifically pass the paths from Clang to get the same ASan runtime as for oneDAL if that is not the system's default compiler:

.. code-block:: bash

    export LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)"

.. note:: This requires both ``clang`` and its runtime libraries to be installed. If using toolkits from ``conda-forge``, then using ``libclang_rt`` requires installing package ``compiler-rt``, in addition to ``clang`` and ``clangxx``. One might also want to install ``llvm-tools`` for enhanced debugging outputs.

Then, the Python memory allocator can be set to ``malloc`` like this:

.. code-block:: bash

    export PYTHONMALLOC=malloc


Putting it all together, the earlier examples building the library in-place and executing a python file with it become as follows:

.. code-block:: bash

    source <path to ASan-enabled oneDAL env.sh>
    CC="ccache icx -fsanitize=address -g" CXX="ccache icpx -fsanitize=address -g" \
        python setup.py build_ext --inplace --force --abs-rpath
    CC="icx -fsanitize=address -g" CXX="icpx -fsanitize=address -g" \
        python setup.py build --abs-rpath
    LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)" \
    PYTHONMALLOC=malloc PYTHONPATH=$(pwd) \
        python <python file.py>

.. note:: Be aware that ASan is known to generate many false-positive reports of memory leaks when used with the |onedal|, NumPy, and SciPy.

Building with other sanitizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UBSan can be used in a similar way as ASan in this library when the |onedal| is built with this sanitizer, by using ``-fsanitize=undefined`` instead, but getting Python to load the required runtime might require using LLD as linker when compiling this library (see argument ``--using-lld`` for more details), and might require loading a different compiler runtime, such as ``libclang_rt.ubsan_standalone-x86_64.so``.
