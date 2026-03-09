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

####################
Quick Start
####################

Get ready to elevate your |sklearn| code with |sklearnex| and experience the benefits of accelerated performance in just a few simple steps.

Compatibility with Scikit-learn*
---------------------------------

|sklearnex| is compatible with the latest stable releases of |sklearn| - see :ref:`software-requirements` for more details.

Integrate |sklearnex|
----------------------

The easiest way to call optimized routines from the |sklearnex| is by patching the ``sklearn`` module
from |sklearn|, which can be achieved with just a call to :obj:`sklearnex.patch_sklearn`:

.. code-block:: python

  from sklearnex import patch_sklearn
  from sklearnex import is_patched_instance
  from sklearn.linear_model import LinearRegression

  is_patched_instance(LinearRegression()) == False

  patch_sklearn()
  from sklearn.linear_model import LinearRegression # now calls sklearnex
  is_patched_instance(LinearRegression()) == True

See :doc:`patching` for more details about patching mechanisms.

Alternatively, functions and classes from the |sklearnex| can be imported directly from the
``sklearnex`` module without patching:

.. code-block:: python

  from sklearnex.linear_model import LinearRegression

Installation
--------------------

.. contents:: :local:

.. tip:: To prevent version conflicts, we recommend creating and activating a new environment for |sklearnex|.

Install from PyPI
**********************

Recommended by default.

To install |sklearnex|, run:

::

  pip install scikit-learn-intelex

.. tip:: Wheels are also available through Intel's index: ``https://software.repos.intel.com/python/pypi``

**Supported Configurations**

.. list-table::
   :align: left

   * - Operating systems
     - Windows*, Linux*
   * - Python versions
     - 3.10, 3.11, 3.12, 3.13, 3.14
   * - Devices
     - CPU, GPU
   * - Modes
     - Single, SPMD (Linux* only)

.. tip:: Running on GPU involves additional dependencies, see :doc:`oneapi-gpu`. SPMD mode has additional requirements on top of GPU ones, see :doc:`distributed-mode` for details.

.. note:: Wheels are only available for x86-64 architecture.

Install through Conda
*********************

To prevent version conflicts, we recommend installing ``scikit-learn-intelex`` into a new conda environment.

.. note::
  The main Anaconda channel also provides distributions of ``scikit-learn-intelex``, but it does not provide the latest versions, nor does
  it provide GPU-enabled builds. It is highly recommended to install ``scikit-learn-intelex`` from either Intel's channel or from conda-forge instead.

.. tabs::

   .. tab:: Intel channel

      Recommended for the Intel® Distribution for Python users.

      To install in a fresh environment: ::

        conda create -n sklex -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels scikit-learn-intelex
        conda activate sklex

      To install in an existing environment: ::

        conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels scikit-learn-intelex

      .. warning::
        Packages from the Intel channel are meant to be used together with dependencies from the **conda-forge** channel, and might not
        work correctly when used in an environment where packages from the ``anaconda`` default channel have been installed. It is
        advisable to use the `miniforge <https://github.com/conda-forge/miniforge>`__ installer for ``conda``/``mamba``, as it comes with
        ``conda-forge`` as the only default channel.

      .. list-table:: **Supported Configurations**
         :align: left

         * - Operating systems
           - Windows*, Linux*
         * - Python versions
           - 3.10, 3.11, 3.12, 3.13, 3.14
         * - Devices
           - CPU, GPU
         * - Modes
           - Single, SPMD (Linux* only)


   .. tab:: Conda-Forge channel

      To install in a fresh environment: ::

        conda create -n sklex -c conda-forge --override-channels scikit-learn-intelex
        conda activate sklex

      To install in an existing environment: ::

        conda install -c conda-forge --override-channels scikit-learn-intelex

      .. list-table:: **Supported Configurations**
         :align: left

         * - Operating systems
           - Windows*, Linux*
         * - Python versions
           - 3.10, 3.11, 3.12, 3.13, 3.14
         * - Devices
           - CPU, GPU
         * - Modes
           - Single, SPMD (Linux* only)

.. tip:: Running on GPU involves additional dependencies, see :doc:`oneapi-gpu`.  SPMD mode has additional requirements on top of GPU ones, see :doc:`distributed-mode` for details.

.. note:: Packages are only available for x86-64 architecture.

.. _build-from-sources:

Build from Sources
**********************

See :doc:`building-from-source` for details.

Install Intel*(R) AI Tools
****************************

Download the Intel AI Tools `here <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html>`_. The extension is already included.

Release Notes
-------------------

See the `Release Notes <https://github.com/uxlfoundation/scikit-learn-intelex/releases>`_ for each version of |sklearnex|.

System Requirements
--------------------

Hardware Requirements
**********************

.. tabs::

   .. tab:: CPU

      Any processor with ``x86-64`` architecture with at least one of the following instruction sets:

        - SSE2
        - SSE4.2
        - AVX2
        - AVX512

      .. note::
        Note: pre-built packages are not provided for other CPU architectures. See :ref:`build-from-sources` for ARM.

   .. tab:: GPU

      - Any Intel® GPU supported by both `DPC++ <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_ and `oneMKL <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-math-kernel-library-system-requirements.html>`_


.. tip:: Read more about hardware comparison in our :ref:`blogs <blogs>`.

.. _software-requirements:

Software Requirements
**********************

.. tabs::

   .. tab:: CPU

      - Linux* OS: Ubuntu* 18.04 or newer
      - Windows* OS 10 or newer
      - Windows* Server 2019 or newer

   .. tab:: GPU

      - A Linux* or Windows* version supported by DPC++ and oneMKL
      - Intel® Compute Runtime (see :ref:`oneapi_gpu`)
      - DPC++ runtime libraries

      .. important::

         If you use accelerators (e.g. GPUs), refer to `oneAPI DPC++/C++ Compiler System Requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_.

|sklearnex| is compatible with the latest stable releases of |sklearn|:

* 1.0.X
* 1.5.X
* 1.6.X
* 1.7.X
* 1.8.X

Memory Requirements
**********************
By default, algorithms in |sklearnex| run in the multi-thread mode. This mode uses all available threads.
Optimized scikit-learn estimators can consume more RAM than their corresponding unoptimized versions.

.. list-table::
   :header-rows: 1
   :align: left

   * - Algorithm
     - Single-thread mode
     - Multi-thread mode
   * - SVM
     - Both |sklearn| and |sklearnex| consume approximately the same amount of RAM.
     - In |sklearnex|, an algorithm with ``N`` threads consumes ``N`` times more RAM.

In all |sklearnex| algorithms with GPU support, computations run on device memory.
The device memory must be large enough to store a copy of the entire dataset.
You may also require additional device memory for internal arrays that are used in computation.


.. seealso::

   :ref:`Samples<samples>`
