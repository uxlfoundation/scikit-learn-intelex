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

Distributed Mode (SPMD)
=======================

|intelex| offers Single Program, Multiple Data (SPMD) supported interfaces for distributed computing.
Several :doc:`GPU-supported algorithms <oneapi-gpu>`
also provide distributed, multi-GPU computing capabilities via integration with |mpi4py|. The prerequisites
match those of GPU computing, along with an MPI backend of your choice (`Intel MPI recommended
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.dcan6r>`_, available
via ``impi_rt`` python package) and the |mpi4py| python package. If using |intelex|
`installed from sources <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-from-sources>`_,
ensure that the spmd_backend is built.

.. important::
  SMPD mode requires the |mpi4py| package used at runtime to be compiled with the same backend as the |intelex|. The PyPI and Conda distributions of |intelex| both use Intel's MPI as backend, and hence require an |mpi4py| also built with Intel's MPI - it can be easily installed from Intel's conda channel as follows::
    
    conda install -c https://software.repos.intel.com/python/conda/ mpi4py

Note that |intelex| now supports GPU offloading to speed up MPI operations. This is supported automatically with
some MPI backends, but in order to use GPU offloading with Intel MPI, set the environment variable ``I_MPI_OFFLOAD`` to ``1`` (providing
data on device without this may lead to a runtime error):

- On Linux*::
    
    export I_MPI_OFFLOAD=1

- On Windows*::
    
    set I_MPI_OFFLOAD=1

Estimators can be imported from the ``sklearnex.spmd`` module. Data should be distributed across multiple nodes as
desired, and should be transfered to a |dpctl| or dpnp array before being passed to the estimator. View a full
example of this process in the |intelex| repository, where many examples of our SPMD-supported estimators are
available: https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/sklearnex/. To run:

- On Linux*::
    
    mpirun -n 4 python linear_regression_spmd.py

- On Windows*::
    
    mpiexec -n 4 python linear_regression_spmd.py

Note that additional mpirun arguments can be added as desired. SPMD-supported estimators are listed in the :ref:`spmd-support` section.

Additionally, daal4py offers some distributed functionality, see
`documentation <https://intelpython.github.io/daal4py/scaling.html>`_ for further details.
