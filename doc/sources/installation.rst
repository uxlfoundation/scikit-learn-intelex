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

=========================
Installation Instructions
=========================

The |sklearnex| can be easily installed as a Python package under name ``scikit-learn-intelex`` from the most
common distribution channels for Python. For GPU support, an additional package ``scikit-learn-intelex-gpu`` is
also made available through the same channels - see :doc:`oneapi-gpu` for more information.

Binary wheels and conda packages are provided for the x86-64 architecture, for both Windows* and Linux*, for the
Python versions supported by current versions of |sklearn| - see :doc:`about` for more information.

Free-threaded Python
--------------------

Source-build support for free-threaded Python starts with CPython 3.14 on
Linux* x86-64 for CPU execution. Importing the native CPU extensions covered by
this support level does not re-enable the GIL. Availability of free-threaded
binary wheels is release-dependent. See :doc:`unsupported` and
:doc:`parallelism` for current platform and execution-mode limitations.

Installing from PyPI
--------------------

To install with ``pip`` from the Python Package Index (PyPI), execute the following command on a terminal: ::

  pip install scikit-learn-intelex

.. tip:: To prevent version conflicts, we recommend creating and activating a new virtual environment for |sklearnex|.

.. tip:: Wheels are also available through Intel's index: ``https://software.repos.intel.com/python/pypi``

Installing from conda-forge
---------------------------

To install the |sklearnex| in a fresh conda environment, execute the following command in a terminal: ::

  conda create -n sklex -c conda-forge --override-channels scikit-learn-intelex
  conda activate sklex

To install in an existing environment: ::

  conda install -c conda-forge --override-channels scikit-learn-intelex

.. hint::
  It is advisable to use the `miniforge <https://github.com/conda-forge/miniforge>`__ installer for ``conda``/``mamba``, as it comes with
  ``conda-forge`` as the default channel.

.. warning::
  The main Anaconda channel also provides distributions of ``scikit-learn-intelex``, but it does not provide the latest versions, nor does
  it provide GPU-enabled builds. It is highly recommended to install ``scikit-learn-intelex`` from either conda-forge or from Intel's channel instead.

.. tip::

  The |sklearnex| is also available at Intel's conda channel: ``https://software.repos.intel.com/python/conda``

  Packages from the Intel channel, which include also optimized versions of NumPy and SciPy, are meant to be used together with
  dependencies from the **conda-forge** channel, and might not work correctly when used in an environment where packages from the
  ``anaconda`` default channel have been installed.

GPU package
-----------

In addition to ``scikit-learn-intelex``, related package ``scikit-learn-intelex-gpu`` is available through the same channels (PyPI, conda-forge, Intel's channels), containing additional features needed to execute operations on GPU through the |sklearnex|.

See :doc:`oneapi-gpu` for details.

Note that ``scikit-learn-intelex-gpu`` is not a standalone package and does not provide additional modules - instead, it extends the same ``sklearnex`` module provided by ``scikit-learn-intelex`` on which it depends, and brings additional dependencies such as the analog GPU package from the |onedal|.

.. _mkl_symbols_note:

Considerations for Python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scikit-learn-intelex-gpu`` package has a transitive runtime dependency on `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__, which includes `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`__ and `LAPACK <https://en.wikipedia.org/wiki/LAPACK>`__ backends that are commonly used by other libraries such as NumPy and SciPy. On Linux*, when ``scikit-learn-intelex-gpu`` is installed, importing the ``sklearnex`` module will trigger loading of oneMKL shared objects into the process together with their symbols (functions) for BLAS and LAPACK.

On PyPI, wheels from major Python packages such as NumPy, SciPy, and PyTorch all bundle their own static-linked versions of BLAS and LAPACK (usually backed by OpenBLAS instead of oneMKL) which avoid possible conflicts with other libraries loading the same symbols on Linux*, while on Anaconda and conda-forge, Python libraries use metapackages for BLAS and LAPACK which ensure only one backend for them is loaded at runtime.

Hence, when it comes to Linux* environments, if all core packages are installed from the same source (either PyPI or conda-forge), there should be no symbol conflicts between different BLAS / LAPACK providers, but if e.g. some packages are installed from PyPI and others are installed from conda-forge, conflicts in symbol resolution might occur, which could manifest for example in thread control for BLAS (e.g. matrix multiplications in NumPy) not working correctly due to calling functions from the wrong vendor, or running times of operations varying depending on which libraries get imported first.

As an alternative, pip-compatible versions of NumPy and SciPy that would use oneMKL as BLAS / LAPACK provider instead of OpenBLAS can be installed as follows:

.. code-block:: bash

    pip install --index-url https://software.repos.intel.com/python/pypi numpy scipy

On Windows*, DLL loading does not bring global symbols into a process in the same way as Linux*, so there should be no potential symbol resolution conflicts from different BLAS / LAPACK providers when using Python libraries.

Typically, other higher-level Python packages such as |sklearn| that make usage of BLAS and LAPACK do so through SciPy's bindings, but if another Python library were to dynamic-link to BLAS and/or LAPACK directly instead, similar considerations should apply.

Building from Sources
---------------------

The |sklearnex| is a fully open-source package (`link to source code <https://github.com/uxlfoundation/scikit-learn-intelex>`_) which
can be compiled from source with standard compiler toolkits.

See :doc:`building-from-source` for details.
