.. Copyright 2021 Intel Corporation
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

.. _distributed:

SPMD (multi-GPU distributed mode)
=================================

|sklearnex| offers Single Program, Multiple Data (SPMD) supported interfaces for distributed computations on multi-GPU setups
(see the :ref:`distributed mode on daal4py <distributed_daal4py>` for distributed algorithms on CPU) **when Running on Linux***.

Several :doc:`GPU-supported algorithms <oneapi-gpu>`
also provide distributed, multi-GPU computing capabilities via integration with |mpi4py|. The prerequisites
match those of GPU computing, along with an MPI backend of your choice (`Intel MPI recommended
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`_, available
via the ``impi_rt`` python/conda package) and the |mpi4py| python package. If using |sklearnex|
`installed from sources <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-from-sources>`_,
ensure that the spmd_backend is built.

.. important::
  SMPD mode requires the |mpi4py| package used at runtime to be compiled with the same MPI backend as the |sklearnex|, or with an ABI-compatible MPI backend. The PyPI and Conda distributions of |sklearnex| are both built with Intel's MPI as backend, which follows the MPICH ABI and hence require an |mpi4py| also built with either Intel's MPI, or with another MPICH-compatible MPI backend (such as MPICH itself) - versions of |mpi4py| built with Intel's MPI can be installed as follows:

    .. tabs::
        .. tab:: From conda-forge
            ::

                conda install -c conda-forge mpi4py mpi=*=impi

        .. tab:: From Intel's conda channel
            ::

                conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels mpi4py mpi=*=impi

            .. warning:: Packages from the Intel channel are meant to be compatible with dependencies from ``conda-forge``, and might not work correctly in environments that have packages installed from the ``anaconda`` channel.

        .. tab:: From Intel's pip Index
            ::

                pip install --index-url https://software.repos.intel.com/python/pypi mpi4py impi_rt

  It also requires the MPI runtime executable (``mpiexec`` / ``mpirun``) to be from the same library that was used to compile |sklearnex| or from a compatible library. Intel's MPI runtime library is offered as a Python package ``impi_rt`` and will be installed together with the ``mpi4py`` package if executing the commands above, but otherwise, it can be installed separately from different distribution channels:

    .. tabs::
        .. tab:: From conda-forge
            ::

                conda install -c conda-forge impi_rt mpi=*=impi

        .. tab:: From Intel's conda channel
            ::

                conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels impi_rt mpi=*=impi

        .. tab:: From PyPI
            ::

                pip install impi_rt

        .. tab:: From Intel's pip Index
            ::

                pip install --index-url https://software.repos.intel.com/python/pypi impi_rt


  Using other MPI backends that are not MPICH-compatible (e.g. OpenMPI) requires building |sklearnex| from source with that backend, and using an |mpi4py| built with that same backend.


Note that |sklearnex| supports GPU offloading to speed up MPI operations. This is supported automatically with
some MPI backends, but in order to use GPU offloading with Intel MPI, it is required to set the environment variable ``I_MPI_OFFLOAD`` to ``1`` (providing
data on device without this may lead to a runtime error): ::

    export I_MPI_OFFLOAD=1

SMPD-aware versions of estimators can be imported from the ``sklearnex.spmd`` module. Data should be distributed across multiple nodes as
desired, and should be transferred to a |dpctl| or `dpnp <https://github.com/IntelPython/dpnp>`__ array before being passed to the estimator.

Note that SPMD estimators allow an additional argument ``queue`` in their ``.fit`` / ``.predict`` methods, which accept :obj:`dpctl.SyclQueue` objects. For example, while the signature for :obj:`sklearn.linear_model.LinearRegression.predict` would be

.. code-block:: python

    def predict(self, X): ...

The signature for the corresponding predict method in ``sklearnex.spmd.linear_model.LinearRegression.predict`` is:

.. code-block:: python

    def predict(self, X, queue=None): ...

Examples of SPMD usage can be found in the GitHub repository for the |sklearnex| under `examples/sklearnex <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/sklearnex>`__.

To run on SPMD mode, first create a python file using SPMD estimators from ``sklearnex.spmd``, such as `linear_regression_spmd.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/sklearnex/linear_regression_spmd.py>`__.

Then, execute the file through MPI under multiple ranks - for example: ::

    mpirun -n 4 python linear_regression_spmd.py

(and remember to set ``I_MPI_OFFLOAD=1`` for Intel's MPI before calling ``mpirun``/``mpiexec``)

Note that additional ``mpirun`` arguments can be added as desired. SPMD-supported estimators are listed in the :ref:`spmd-support` section.
