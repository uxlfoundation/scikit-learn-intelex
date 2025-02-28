.. Copyright 2020 Intel Corporation
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

###############################################
Scaling on Distributed Memory (Multiprocessing)
###############################################

.. include:: note.rst

It's Easy
---------
daal4py operates in SPMD style (Single Program Multiple Data), which means your
program is executed on several processes (e.g. similar to MPI).  The use of MPI is
not required for daal4py's SPMD-mode to work, all necessary communication and
synchronization happens under the hood of daal4py. It is possible to use daal4py and
mpi4py in the same program, though.

Only very minimal changes are needed to your daal4py code to allow daal4py to
run on a cluster of workstations. Initialize the distribution engine::

  daalinit()

Add the distribution parameter to the algorithm construction::

  kmi = kmeans_init(10, method="plusPlusDense", distributed=True)

When calling the actual computation each process expects an input file or input
array/DataFrame. Your program needs to tell each process which
file/array/DataFrame it should operate on. Like with other SPMD programs this is
usually done conditionally on the process id/rank ('daal4py.my_procid()'). Assume
we have one file for each process, all having the same prefix 'file' and being
suffixed by a number. The code could then look like this::

  result = kmi.compute("file{}.csv", daal4py.my_procid())

The result of the computation will now be available on all processes.

Finally stop the distribution engine::

  daalfini()

That's all for the python code::

  from daal4py import daalinit, daalfini, kmeans_init
  daalinit()
  kmi = kmeans_init(10, method="plusPlusDense", distributed=True)
  result = kmi.compute("file{}.csv", daal4py.my_procid())
  daalfini()

To actually get it executed on several processes use standard MPI mechanics,
like::

  mpirun -n 4 python ./kmeans.py

The binaries provided by Intel use the Intel® MPI library, but
daal4py can also be compiled for any other MPI implementation.

Supported Algorithms and Examples
---------------------------------
The following algorithms support distribution:

- PCA (pca)

  - `PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_spmd.py>`_

- SVD (svd)

  - `SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_spmd.py>`_

- Linear Regression Training (linear_regression_training)

  - `Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_spmd.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_spmd.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_spmd.py>`_

- K-Means (kmeans_init and kmeans)

  - `K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans_spmd.py>`_

- Correlation and Variance-Covariance Matrices (covariance)

  - `Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_spmd.py>`_

- Moments of Low Order (low_order_moments)

  - `Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_spmd.py>`_

- QR Decomposition (qr)

  - `QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr_spmd.py>`_
