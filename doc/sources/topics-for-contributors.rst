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

=======================
Topics for Contributors
=======================

Adding an estimator
-------------------

Estimator classes in the |sklearnex| are wrappers over algorithms from the |onedal|. In order to add a new estimator, an example class ``DummyEstimator`` is available in the library, along with code comments and tests which explain how it should work. Estimators spawn multiple files, ranging from C++ wrappers from PyBind11, direct wrappers in the ``onedal/`` module, scikit-learn-conformant wrappers over those in the ``sklearnex/`` module, direct tests, configurations for general tests, and others.

Example estimator
~~~~~~~~~~~~~~~~~

The following files and folders might be of help when looking at how the example ``DummyEstimator`` works and what is needed of an estimator:

- Files under folder `onedal/dummy/ <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/onedal/dummy>`__.
- Files under folder `sklearnex/dummy/ <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/sklearnex/dummy>`__.

The following files might also require changes after adding a new estimator - look out for the "dummy" keyword:

- Import-related files:

    - `onedal/dal.cpp <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/onedal/dal.cpp>`__.
    - `onedal/__init__.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/onedal/__init__.py>`__.
    - `setup.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/setup.py>`__.
    - `sklearnex/__init__.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/__init__.py>`__.
    - `sklearnex/dispatcher.py.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/dispatcher.py.py>`__.

- Test-related files:

    - `sklearnex/tests/utils/base.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/utils/base.py>`__.
    - `sklearnex/tests/test_common.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/test_common.py>`__.
    - `sklearnex/tests/test_memory_usage.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/test_memory_usage.py>`__.
    - `sklearnex/tests/test_n_jobs_support.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/test_n_jobs_support.py>`__.
    - `sklearnex/tests/test_patching.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/test_patching.py>`__.
    - `sklearnex/tests/test_run_to_run_stability.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/sklearnex/tests/test_run_to_run_stability.py>`__.
    - `.ci/scripts/select_sklearn_tests.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/.ci/scripts/select_sklearn_tests.py>`__.

.. note:: The library contains lots of classes with legacy code from previous designs that do not work in the same way as the ``DummyEstimator`` class, such as classes based off ``daal4py``. New estimators should nevertheless not try to mimic those, and follow instead the design from ``DummyEstimator``.

.. tip:: Another good reference example for how estimators should be implemented is :obj:`sklearn.linear_model.LinearRegression` from the ``sklearnex`` module.

For estimators that somehow depend on functionality that is only exposed through ``daal4py``, an internal wrapper akin to the files under ``onedal/`` must first be created under ``daal4py/sklearn``, and then imported in a corresponding class on ``onedal/``. Note that new functionalities in the |onedal| are meant to be introduced through the oneAPI interface, so only legacy functionalities should ever need to go through this route.

Version compatibilities
-----------------------

OneDAL
~~~~~~

The |sklearnex| is intended to be backwards-compatible with different versions of the |onedal|, but not forwards-compatible except within a major release series - meaning: it is meant to run with a version of the |onedal| that is lower or equal than the version of the |sklearnex|, such that ``onedal==2025.0`` + ``sklearnex==2025.0`` and ``onedal==2025.0`` + ``sklearnex==2025.2`` should both work correctly, even though the latter might not expose the same functionalities with ``onedal==2025.0`` as with ``onedal==2025.2``.

This is achieved with conditional runtime checks of the library versions in order to determine whether some class or function or similar should be defined or not, through the provided function ``daal_check_version``, which accepts a tuple as argument containing the major version number, the ``"P"`` string (other possibilities for this parameter are not used anymore), and the minor version **multiplied by 100**. So for example, if a given piece of code requires ``onedal>=2025.2``, the function should be called as follows:

.. code-block:: python

    if daal_check_version((2025, "P", 200)):
        # code branch for onedal>=2025.2
    else:
        # code branch for onedal<2025.2

.. hint:: This helper is meant for usage in both source code and tests.

On C++ code, the macro ``ONEDAL_VERSION`` should be checked at compile-time for conditional code inclusions or exclusions. This macro contains a single integral number with the major version, followed by the minor version using 2 digits, and other patch versions using another two digits. For example, if a given piece of code requires ``onedal>=2025.2``, the check would be as follows:

.. code-block:: cpp

    #if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20250200
    // code for newer version
    #else
    // code for older version
    #endif

Scikit-learn
~~~~~~~~~~~~

The |sklearnex| is intended to be compatible with multiple versions of |sklearn|. In order to achieve this compatibility, conditional runtime checks for the version of |sklearn| are executed in order to offer different code paths for different versions, through function ``sklearn_check_version``, which accepts a string with the major and minor version as recognized by ``pip``. For example, in order to have different code branches depending on ``sklearn>=1.7`` (which would also trigger for ``sklearn==1.7.2``, for example), the following can be used:

.. code-block:: python

    if sklearn_check_version("1.7"):
        # code branch for sklearn>=1.7
    else:
        # code branch for sklearn<1.7

Test helpers
------------

Note that not all estimators offer the same functionalities, and thus tests should be designed accordingly. The tests provide some custom marks, fixtures, and helpers that one might to use for some cases:

- ``@pytest.mark.allow_sklearn_fallback``: will avoid having tests fail when they end up calling procedures from |sklearn| instead of from the |onedal|. This can be helpful for example when testing that some corner case falls back correctly when it should.
- ``onedal.tests.utils._dataframes_support._as_numpy``: this function can be used to convert an input array or data frame to NumPy, regardless of whether it lives on host or on device, and regardless of array API support.
- ``pass_if_not_implemented_for_gpu``: skips tests not implemented for GPU when GPU support is enabled. Requires a skip reason argument that matches the backend's error message.

Tests with optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests that require optional dependencies in order to execute should have a conditional skip logic through usage of ``@pytest.mark.skipif``. The test files are meant to be executable without the optional dependencies being installed, so they should be imported conditionally or in a ``try`` + ``except ImportError`` block.

SPMD tests
~~~~~~~~~~

Tests that involve distributed mode functionalities should rely on ``pytest-mpi`` and need to be marked with ``@pytest.mark.mpi``.

Running benchmarks
------------------

As this library aims to offer accelerated versions of algorithms, when it comes to adding or modifying estimators and related helper functions, it is usually helpful - and in many cases required - to conduct benchmarks to assess the performance implications of changes, whether against |sklearn| or against the current version of the |sklearnex| when introducing changes.

Benchmarks are usually conducted through the `scikit-learn_bench <https://github.com/IntelPython/scikit-learn_bench>`__ tool, which lives in a different repository. See the instructions in that repository for how to run the appropriate benchmarks.

Results from benchmarks are usually shared as a relative improvement over the baseline being compared against, which will be available in the sheets of the generated ``.xlsx`` comparison reports from that repository. Usually, the geometric mean is used as a final number, but changes for individual datasets and estimator methods are typically still of interest within a given pull request.

Building the documentation
--------------------------

The source code for the documentation being rendered here is available from the same repository as the library's source code, and hosted on GitHub pages through automated deployments. The source code for the documentation is written in Sphinx, taking some docstrings from the classes and functions in the library to render them.

Thus, building the documentation from source requires being able to import the library in the same Python environment that is building the documentation, in addition to having all of the Python packages used by the Sphinx built script, such as Sphinx itself and the Sphinx extensions used throughout these docs.

Building documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development purposes, it's helpful to build the docs locally to inspect them offline without deploying, based off the current version of the source code instead of a public release version. This can be done using the provided scripts in this repository.

Requirements
************

Being based off Sphinx, the scripts for building documentation require a Python environment with documentation-related packages installed. The locked requirements (and note that in many cases specific versions of the dependencies might be needed) are available in file `requirements-doc.txt <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/requirements-doc.txt>`__. They can be installed from the root of the repository as follows:

.. code-block:: bash

    pip install -r requirements-doc.txt

.. tip:: It's advised to create a separate Python environment for building the docs due to the locked requirements and version conflicts with what's used for the tests.

Instructions
************

With the necessary dependencies being installed, the docs can then be built locally **on Linux\*** by executing the following script **from the root of the repository**:

.. code-block:: bash

    ./doc/build-doc.sh

.. note:: The script accepts additional arguments and environment variables which are used for the versioned doc pages hosted on GitHub pages. Those are not meant to be used for local development.

The script will copy over necessary files to the docs folder and make calls to Sphinx to build the docs as HTML. After that script is executed for the first time, if no new embedded notebooks / examples  from ``.py`` files have been added, the docs can be built without the script using the provided ``Makefile``:

.. code-block:: bash

    cd doc
    make clean
    make html

.. note:: The docs can be built on Windows* using the file ``make.bat``, but be aware that it will not render everything correctly if the commands from ``build-doc.sh`` that copy files haven't been executed.

Copyright headers
-----------------

Each new file added to the project must include the following copyright notice - note that this project is closely tied to the |onedal| and hence shares the same copyright header. The following copyright headers should be used:

- For Python and YAML files:

    .. code-block:: python

        # ==============================================================================
        # Copyright contributors to the oneDAL project
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        # ==============================================================================

- For C++ files:

    .. toggle::

        .. code-block:: cpp

            /*
             * Copyright contributors to the oneDAL project
             *
             * Licensed under the Apache License, Version 2.0 (the "License");
             * you may not use this file except in compliance with the License.
             * You may obtain a copy of the License at
             *
             *     http://www.apache.org/licenses/LICENSE-2.0
             *
             * Unless required by applicable law or agreed to in writing, software
             * distributed under the License is distributed on an "AS IS" BASIS,
             * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
             * See the License for the specific language governing permissions and
             * limitations under the License.
             */

- For markdown files:

    .. toggle::

        .. code-block::

            <!--
            ********************************************************************************
            * Copyright contributors to the oneDAL project
            *
            * Licensed under the Apache License, Version 2.0 (the "License");
            * you may not use this file except in compliance with the License.
            * You may obtain a copy of the License at
            *
            *     http://www.apache.org/licenses/LICENSE-2.0
            *
            * Unless required by applicable law or agreed to in writing, software
            * distributed under the License is distributed on an "AS IS" BASIS,
            * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            * See the License for the specific language governing permissions and
            * limitations under the License.
            *******************************************************************************/-->

- For javascript files:

    .. toggle::

        .. code-block:: javascript

            // Copyright contributors to the oneDAL project
            //
            // Licensed under the Apache License, Version 2.0 (the "License");
            // you may not use this file except in compliance with the License.
            // You may obtain a copy of the License at
            //
            //     http://www.apache.org/licenses/LICENSE-2.0
            //
            // Unless required by applicable law or agreed to in writing, software
            // distributed under the License is distributed on an "AS IS" BASIS,
            // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            // See the License for the specific language governing permissions and
            // limitations under the License.

- For rst files:

    .. toggle::

        .. code-block::

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

If for some reason it doesn't make sense to include this copyright header in a text-based file (e.g. json files), said file needs to be added to the `exclusion list <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/.github/.licenserc.yaml>`__, but this should be a rare occurrence.
