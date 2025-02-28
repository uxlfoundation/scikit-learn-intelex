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

Get ready to elevate your |sklearn| code with |intelex| and experience the benefits of accelerated performance in just a few simple steps.

Compatibility with Scikit-learn*
---------------------------------

|intelex| is compatible with the latest stable releases of |sklearn| - see :ref:`software-requirements` for more details.

Integrate |intelex|
--------------------

Patching
**********************

Once you install the |intelex|, you can replace estimator classes (algorithms) that exist in the ``sklearn`` module from |sklearn| with their optimized versions from the extension.
This action is called `patching`. This is not a permanent change so you can always undo the patching if necessary.

To patch |sklearn| with the |intelex|, the following methods can be used:

.. list-table::
   :header-rows: 1
   :align: left

   * - Method
     - Action
   * - Use a flag in the command line
     - Run this command:

       ::

          python -m sklearnex my_application.py
   * - Modify your script
     - Add the following lines:

       ::

          from sklearnex import patch_sklearn
          patch_sklearn()
   * - Import an estimator from the ``sklearnex`` module
     - Run this command:

       ::

          from sklearnex.neighbors import NearestNeighbors



These patching methods are interchangeable.
They support different enabling scenarios while producing the same result.


**Example**

This example shows how to patch |sklearn| by modifing your script. To make sure that patching is registered by the scikit-learn estimators, always import module ``sklearn`` after these lines.

.. code-block:: python
  :caption: Example: Drop-In Patching

    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    # You need to re-import scikit-learn algorithms after the patch
    from sklearn.cluster import KMeans

    # The use of the original Scikit-learn is not changed
    X = np.array([[1,  2], [1,  4], [1,  0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"kmeans.labels_ = {kmeans.labels_}")


Global Patching
**********************

You can also use global patching to patch all your |sklearn| applications without any additional actions.

Before you begin, make sure that you have read and write permissions for Scikit-learn files.

With global patching, you can:

.. list-table::
   :header-rows: 1
   :align: left

   * - Task
     - Action
     - Note
   * - Patch all supported algorithms
     - Run this command:

       ::

          python -m sklearnex.glob patch_sklearn

     - If you run the global patching command several times with different parameters, then only the last configuration is applied.
   * - Patch selected algorithms
     - Use ``--algorithm`` or ``-a`` keys with a list of algorithms to patch. For example, to patch only ``SVC`` and ``RandomForestClassifier`` estimators, run

       ::

           python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier

     -
   * - Enable global patching via code
     - Use the ``patch_sklearn`` function with the ``global_patch`` argument:

       ::

          from sklearnex import patch_sklearn
          patch_sklearn(global_patch=True)
          import sklearn

     - After that, Scikit-learn patches is enabled in the current application and in all others that use the same environment.
   * - Disable patching notifications
     - Use ``--no-verbose`` or ``-nv`` keys:

       ::

          python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier -nv
     -
   * - Disable global patching
     - Run this command:

       ::

          python -m sklearnex.glob unpatch_sklearn
     -
   * - Disable global patching via code
     - Use the ``global_patch`` argument in the ``unpatch_sklearn`` function

       ::

          from sklearnex import unpatch_sklearn
          unpatch_sklearn(global_patch=True)
     -

.. tip:: If you clone an environment with enabled global patching, it will already be applied in the new environment.

Unpatching
**********************

To undo the patch (also called `unpatching`) is to return the ``sklearn`` module to the original implementation and
replace patched estimators with the stock |sklearn| estimators.

To unpatch successfully, you must reimport the ``sklearn`` module(s)::

  sklearnex.unpatch_sklearn()
  # Re-import scikit-learn algorithms after the unpatch
  from sklearn.cluster import KMeans


Installation
--------------------

.. contents:: :local:

.. tip:: To prevent version conflicts, we recommend creating and activating a new environment for |intelex|.

Install from PyPI
**********************

Recommended by default.

To install |intelex|, run:

::

  pip install scikit-learn-intelex

**Supported Configurations**

.. list-table::
   :align: left

   * - Operating systems
     - Windows*, Linux*
   * - Python versions
     - 3.9, 3.10, 3.11, 3.12, 3.13
   * - Devices
     - CPU, GPU
   * - Modes
     - Single, SPMD

.. tip:: Running on GPU involves additional dependencies, see :doc:`oneapi-gpu`. SPMD mode has additional requirements on top of GPU ones, see :doc:`distributed-mode` for details.

.. note:: Wheels are only available for x86-64 architecture.

Install from Anaconda* Cloud
********************************************

To prevent version conflicts, we recommend installing `scikit-learn-intelex` into a new conda environment.

*Note: the main Anaconda channel also provides distributions of scikit-learn-intelex, but it does not provide the latest versions, nor does it provide GPU-enabled builds. It is highly recommended to install it from either Intel's channel or conda-forge instead.*

.. tabs::

   .. tab:: Intel channel

      Recommended for the Intel® Distribution for Python users.

      To install, run::

        conda install -c https://software.repos.intel.com/python/conda/ scikit-learn-intelex

      .. list-table:: **Supported Configurations**
         :align: left

         * - Operating systems
           - Windows*, Linux*
         * - Python versions
           - 3.9, 3.10, 3.11, 3.12, 3.13
         * - Devices
           - CPU, GPU
         * - Modes
           - Single, SPMD


   .. tab:: Conda-Forge channel

      To install, run::

        conda install -c conda-forge scikit-learn-intelex

      .. list-table:: **Supported Configurations**
         :align: left

         * - Operating systems
           - Windows*, Linux*
         * - Python versions
           - 3.9, 3.10, 3.11, 3.12, 3.13
         * - Devices
           - CPU, GPU
         * - Modes
           - Single, SPMD

.. tip:: Running on GPU involves additional dependencies, see :doc:`oneapi-gpu`.  SPMD mode has additional requirements on top of GPU ones, see :doc:`distributed-mode` for details.

.. note:: Packages are only available for x86-64 architecture.

.. _build-from-sources:

Build from Sources
**********************

See `Installation instructions <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-from-sources>`_ to build |intelex| from the sources.

Install Intel*(R) AI Tools
****************************

Download the Intel AI Tools `here <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html>`_. The extension is already included.

Release Notes
-------------------

See the `Release Notes <https://github.com/uxlfoundation/scikit-learn-intelex/releases>`_ for each version of |intelex|.

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


.. tip:: Intel(R) processors provide better performance than other CPUs. Read more about hardware comparison in our :ref:`blogs <blogs>`.

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
      - Intel® GPGPU drivers
      - DPC++ runtime libraries

      .. important::

         If you use accelerators (e.g. GPUs), refer to `oneAPI DPC++/C++ Compiler System Requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_.

|intelex| is compatible with the latest stable releases of |sklearn|:

* 1.0.X
* 1.1.X
* 1.2.X
* 1.3.X
* 1.4.X
* 1.5.X
* 1.6.X

Memory Requirements
**********************
By default, algorithms in |intelex| run in the multi-thread mode. This mode uses all available threads.
Optimized scikit-learn estimators can consume more RAM than their corresponding unoptimized versions.

.. list-table::
   :header-rows: 1
   :align: left

   * - Algorithm
     - Single-thread mode
     - Multi-thread mode
   * - SVM
     - Both |sklearn| and |intelex| consume approximately the same amount of RAM.
     - In |intelex|, an algorithm with ``N`` threads consumes ``N`` times more RAM.

In all |intelex| algorithms with GPU support, computations run on device memory.
The device memory must be large enough to store a copy of the entire dataset.
You may also require additional device memory for internal arrays that are used in computation.


.. seealso::

   :ref:`Samples<samples>`
