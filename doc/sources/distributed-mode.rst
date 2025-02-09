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

.. _distributed:

Distributed Mode
================

|intelex| offers Single Program, Multiple Data (SPMD) supported interfaces for distributed computing.
Several `GPU-supported algorithms <https://uxlfoundation.github.io/scikit-learn-intelex/latest/oneapi-gpu.html#>`_
also provide distributed, multi-GPU computing capabilities via integration with ``mpi4py``. The prerequisites
match those of GPU computing, along with an MPI backend of your choice (`Intel MPI recommended
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.dcan6r>`_, available
via ``impi-devel`` python package) and the ``mpi4py`` python package. If using |intelex|
`installed from sources <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-from-sources>`_,
ensure that the spmd_backend is built.

Note that |intelex| now supports GPU offloading to speed up MPI operations. This is supported automatically with
some MPI backends, but in order to use GPU offloading with Intel MPI, set the following environment variable (providing
data on device without this may lead to a runtime error):

::

     export I_MPI_OFFLOAD=1

Estimators can be imported from the ``sklearnex.spmd`` module. Data should be distributed across multiple nodes as
desired, and should be transfered to a dpctl or dpnp array before being passed to the estimator. View a full
example of this process in the |intelex| repository, where many examples of our SPMD-supported estimators are
available: https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/sklearnex/. To run:

::

  mpirun -n 4 python linear_regression_spmd.py

Note that additional mpirun arguments can be added as desired. SPMD-supported estimators are listed in the
`algorithms support documentation <https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html#spmd-support>`_.

Additionally, daal4py offers some distributed functionality, see
`documentation <https://intelpython.github.io/daal4py/scaling.html>`_ for further details.
