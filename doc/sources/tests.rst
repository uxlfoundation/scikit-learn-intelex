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

=============
Running Tests
=============

Overview
--------

The |sklearnex| contains a test suite consisting of a mixture of smoke tests around patching along with unit tests, which are written in a mixture of Python's ``unittest`` (for legacy interfaces) and ``pytest``. However, all of the tests are executed with ``pytest`` as runner. Apart from the tests, code examples are also executed, but are not thoroughly checked for correctness, just for executing without erroring out.

Running test scripts
--------------------

Requirements
~~~~~~~~~~~~

As the library is designed with optional components and integrates with external packages that are optional by design, executing the tests involves additional dependencies, some of which are mandatory and some of which are optional.

The mandatory dependencies for tests with locked versions of packages are listed under a file `requirements-test.txt <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/requirements-test.txt>`__, but it is not strictly necessary to have the exact versions listed there. Those versioned requirements can be installed with ``pip`` as follows:

.. code-block:: bash

    pip install -r requirements-test.txt

Some tests will only execute depending on the availability of optional dependencies at runtime. Other optional dependencies that will trigger additional tests can be installed as follows, assuming a Linux* system:

.. code-block:: bash

    pip install \
        dpctl `# for GPU functionalities` \
        dpnp `# for array API and GPU functionalities`

    pip install --index-url https://software.repos.intel.com/python/pypi \
        torch `# for array API`

    pip install --index-url https://software.repos.intel.com/python/pypi \
        mpi4py impi_rt `# for distributed mode, be sure to install from Intel's index` \
        && pip install pytest-mpi `# also required, but not from Intel's index`

.. warning:: It might not be possible to install all of the test dependencies simultaneously in the same Python environment. In particular, dependencies of ``torch`` and ``dpctl`` / ``dpnp`` are potentially incompatible if using pre-built distributions. Try using different Python environments for each set of dependencies to test.

.. warning:: If installing dependencies for distributed mode from ``pip``, be sure to install ``mpi4py`` from the Intel ``pip`` index to ensure that it uses a compatible MPI backend. See :doc:`distributed-mode` for details.

Executing tests
~~~~~~~~~~~~~~~

In order to run the whole test suite, the following script can be used on Linux*:

.. code-block:: bash

    conda-recipe/run_test.sh

.. warning:: This script must be executed from the root of the repository.

It also comes with an analog for Windows*:

.. code-block:: console

    call conda-recipe/run_test.bat

Individual test files or tests can be executed with PyTest under different options (such as a different verbosity mode, stopping at the first failure, etc.) with calls to pytest - for example:

.. code-block:: bash

    pytest sklearnex/ensemble/tests/test_forest.py

.. hint:: If executing these from the root of the repository, there might be naming clashes between the folders and the installed Python modules. It might be helpful to :ref:`build the library extensions in-place <build_script>` and set ``$PYTHONPATH`` to avoid problems.

Configurable options
~~~~~~~~~~~~~~~~~~~~

The files ``run_test.sh`` and ``run_test.bat`` offer configurable behaviors through environment variables and command line arguments:

- Environment variable ``$NO_DIST``, if set, will avoid running distributed mode tests. Note that executing these tests requires additional dependencies, otherwise they will be skipped either way.
- Environment variable ``$PYTHON`` can be used to set a Python interpreter under an MPI runner to execute distributed tests on Windows* - for example: ``set "PYTHON=mpiexec -n 2 python"``. **This variable is required for distributed mode tests on Windows\*** - if not set, ``NO_DIST`` will be automatically set to 1.

    - On Linux*, this same variable can be used to set the Python interpreter that will run the tests for patching functionality.
- Passing argument ``--json-report`` will generate JSON reports of each test component under path ``/.pytest_reports``. Note that, if the folder is not empty, existing files will be deleted.
- Environment variable ``$COVERAGE_RCFILE``, if set, will make it generate coverage reports under the path specified from this variable.

Running distributed mode examples
---------------------------------

A helper script `tests/run_examples.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/tests/run_examples.py>`__ is provided for executing the `code examples <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples>`__ for distributed mode on both GPU (see :doc:`distributed-mode`) and CPU (see :doc:`distributed_daal4py`).

This script is not executed as part of the regular test suite, even though the examples might be executed in non-distributed mode during execution of other scripts.

Executing these distributed mode examples requires all of the optional dependencies for distributed mode tests. With those installed, the script can be executed as follows:

.. code-block:: bash

    python tests/run_examples.py

.. warning:: This script needs to be executed from the root of the repository as working directory. The script will modify the working directory when it launches subprocesses, so if using environment variables like ``$PYTHONPATH``, these need to be set as absolute paths (not relative).

.. _conformance_tests:

Scikit-learn's test suite
-------------------------

The |sklearnex| is regularly tested for correctness through the test suite of |sklearn| itself executed with patching applied, referred throughout the CI jobs and files as 'conformance testing'.

Executing tests
~~~~~~~~~~~~~~~

To execute the |sklearn| conformance tests, the following script can be used:

.. code-block:: bash

    ./.ci/scripts/run_sklearn_tests.sh


Note that some tests are known to produce failures - for example, :obj:`sklearn.linear_model.LinearRegression` allows an argument ``copy_X``, and one of its tests checks that passing ``copy_X=False`` modifies the 'X' input in-place, while the |sklearnex| never modifies this data regardless of the argument ``copy_X``, hence the test would show a failure under a patched call to |sklearn|, even though the results do not change.

Cases that are known to fail are not executed during these conformance test. The list of deselected tests can be found under `deselected_tests.yaml <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/deselected_tests.yaml>`__.

Selecting tests
***************

Individual tests can be executed through the underlying ``.py`` file that the ``.sh`` script executes, and other custom selections or deselections can be changed on-the-fly there through usage of environment variables - for example:

.. code-block:: bash

    SELECTED_TESTS=all DESELECTED_TESTS="" python .ci/scripts/run_sklearn_tests.py

The environment variables ``SELECTED_TESTS`` and ``DESELECTED_TESTS`` accept space-separated names of tests from the test suite of |sklearn|, as PyTest would take them if executed from the root of the repository. For example, in order to execute the test named `test_classification_toy <https://github.com/scikit-learn/scikit-learn/blob/0c27a07f68e0eda7e1fcbce44a7615addec7f232/sklearn/ensemble/tests/test_forest.py#L122C5-L122C28>`__ from the file ``ensemble/tests/test_forest.py`` `from the scikit-learn repository <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/tests/test_forest.py>`__, the following can be used:

.. code-block:: bash

    SELECTED_TESTS="ensemble/tests/test_forest.py::test_classification_toy" DESELECTED_TESTS="" \
        python .ci/scripts/run_sklearn_tests.py

Note that these are passed to the ``pytest`` call, so other forms of pattern matching accepted by PyTest can also be used.


.. note:: If building the extension modules in-place :ref:`per the instructions here <build_script>`, it requires also setting ``$PYTHONPATH`` for this script to work.

Further arguments to pytest can be supplied by passing them as arguments to the `.py` runner - for example:

.. code-block:: bash

    SELECTED_TESTS=all DESELECTED_TESTS="" python .ci/scripts/run_sklearn_tests.py -x

GPU mode
********

The tests can also be made to run on GPU, either by passing argument ``gpu`` to ``run_sklearn_tests.sh``, or by passing argument ``--device <device name>`` to  ``run_sklearn_tests.py`` - example:

.. code-block:: bash

    ./.ci/scripts/run_sklearn_tests.sh gpu

Preview mode
************

Note that :doc:`preview mode <preview>` is not tested by default - in order to test it, it's necessary to set environment variable ``SKLEARNEX_PREVIEW=1`` to enable patching of such functionalities before executing either of these scripts (``.sh`` / ``.py``). The ``.sh`` script by default will take care of deselecting tests that require preview mode for patching when this environment variable is not set.

Producing a test report
~~~~~~~~~~~~~~~~~~~~~~~

Optionally, a JSON report of the results can be produced (requires package ``pytest-json-report``) by setting an environment variable ``JSON_REPORT_FILE``, indicating the location where to produce a JSON output file - note that the test runner changes the PyTest root directory, so it should be specified as an absolute path, or otherwise will get written into the ``site-packages`` folder for ``sklearn``:

.. code-block:: bash

    SELECTED_TESTS=all \
    DESELECTED_TESTS="" \
    JSON_REPORT_FILE="$(pwd)/sklearn_test_results.json" \
        python .ci/scripts/run_sklearn_tests.py


Comparing test reports
**********************

A small utility to compare two JSON test reports is provided under `tests/util_compare_json_reports.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/tests/util_compare_json_reports.py>`__, which can be useful for example when comparing changes before and after a given commit.

The file is a python script which produces a new JSON output file highlighting the tests that had different outcomes between two JSON reports. It needs to be executed with the following arguments, prefixed with two dashes and with the value passed after an equal sign (e.g. ``--arg1=value``):

- ``json1``: path to a first JSON report file from ``pytest-json-report``.
- ``json2``: path to a second JSON report file from ``pytest-json-report``.
- ``name1``: name that the tests from the first file will use as JSON keys in the generated output file.
- ``name2``: name that the tests from the second file will use as JSON keys in the generated output file.
- ``output``: file name where to save the result JSON file that highlights the differences.

Example:

.. code-block:: bash

    python tests/util_compare_json_reports.py \
        --json1=logs_before.json \
        --json2=logs_after.json \
        --name1="before" \
        --name2="after" \
        --output="diffs_before_after.json"


The result will be a new JSON file which will contain only entries for tests that were present in both files and which had different outcomes, with a structure as follows:

.. code-block::

    "test_name": { # taken from 'nodeid' in the pytest json reports
        <name1>: { # taken from argument 'name1'
            ...    # json from entry in pytest report under 'tests', minus key 'nodeid'
        },
        <name2>: { # taken from argument 'name2'
            ...    # json from entry in pytest report under 'tests', minus key 'nodeid'
        }
    }
