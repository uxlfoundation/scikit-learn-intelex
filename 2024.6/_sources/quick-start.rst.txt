.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. |intelex_repo| replace:: |intelex| repository
.. _intelex_repo: https://github.com/intel/scikit-learn-intelex

####################
Quick Start
####################

Get ready to elevate your scikit-learn code with |intelex| and experience the benefits of accelerated performance in just a few simple steps. 

Compatibility with Scikit-learn*
---------------------------------

Intel(R) Extension for Scikit-learn is compatible with the last four versions of scikit-learn.

Integrate |intelex|
--------------------

Patching 
**********************

Once you install Intel*(R) Extension for Scikit-learn*, you replace algorithms that exist in the scikit-learn package with their optimized versions from the extension. 
This action is called ``patching``. This is not a permanent change so you can always undo the patching if necessary.

To patch Intel® Extension for Scikit-learn, use one of these methods: 

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

This example shows how to patch Intel(R) extension for Scikit-Learn by modifing your script. To make sure that patching is registered by the scikit-learn estimators, always import scikit-learn after these lines.
  
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

You can also use global patching to patch all your scikit-learn applications without any additional actions.

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

To undo the patch (also called `unpatching`) is to return scikit-learn to original implementation and
replace patched algorithms with the stock scikit-learn algorithms.

To unpatch successfully, you must reimport the scikit-learn package::

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
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.8
     - Python 3.9
     - Python 3.10
     - Python 3.11
     - Python 3.12
   * - Linux* OS
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
   * - Windows* OS
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]



Install from Anaconda* Cloud
********************************************

To prevent version conflicts, we recommend installing `scikit-learn-intelex` into a new conda environment.

.. tabs::

   .. tab:: Conda-Forge channel

      Recommended by default. 
      
      To install, run::

        conda install scikit-learn-intelex -c conda-forge
      
      .. list-table:: **Supported Configurations**
         :header-rows: 1
         :align: left

         * - OS / Python version
           - Python 3.8
           - Python 3.9
           - Python 3.10
           - Python 3.11
           - Python 3.12
         * - Linux* OS
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
         * - Windows* OS
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]


   .. tab:: Intel channel

      Recommended for the Intel® Distribution for Python users. 

      To install, run::

        conda install scikit-learn-intelex -c intel
      
      .. list-table:: **Supported Configurations**
         :header-rows: 1
         :align: left

         * - OS / Python version
           - Python 3.8
           - Python 3.9
           - Python 3.10
           - Python 3.11
           - Python 3.12
         * - Linux* OS
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
         * - Windows* OS
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
           - [CPU, GPU]
 


   .. tab:: Main channel

      To install, run::

        conda install scikit-learn-intelex
      
      .. list-table:: **Supported Configurations**
         :header-rows: 1
         :align: left

         * - OS / Python version
           - Python 3.8
           - Python 3.9
           - Python 3.10
           - Python 3.11
           - Python 3.12
         * - Linux* OS
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
         * - Windows* OS
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]
           - [CPU]



Build from Sources
**********************

See `Installation instructions <https://github.com/intel/scikit-learn-intelex/blob/main/INSTALL.md>`_ to build |intelex| from the sources.

Install Intel*(R) AI Tools
****************************

Download the Intel AI Tools `here <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html>`_. The extension is already included.

Release Notes
-------------------

See the `Release Notes <https://github.com/intel/scikit-learn-intelex/releases>`_ for each version of Intel® Extension for Scikit-learn*.  

System Requirements 
--------------------

Hardware Requirements
**********************

.. tabs::

   .. tab:: CPU

      All processors with ``x86`` architecture with at least one of the following instruction sets:

        - SSE2
        - SSE4.2
        - AVX2
        - AVX512
       
      .. note:: ARM* architecture is not supported.

   .. tab:: GPU

      - All Intel® integrated and discrete GPUs
      - Intel® GPU drivers


.. tip:: Intel(R) processors provide better performance than other CPUs. Read more about hardware comparison in our :ref:`blogs <blogs>`.


Software Requirements
**********************

.. tabs::

   .. tab:: CPU

      - Linux* OS: Ubuntu* 18.04 or newer
      - Windows* OS 10 or newer
      - Windows* Server 2019 or newer

   .. tab:: GPU

      - Linux* OS: Ubuntu* 18.04 or newer
      - Windows* OS 10 or newer
      - Windows* Server 2019 or newer
      
      .. important::
         
         If you use accelerators, refer to `oneAPI DPC++/C++ Compiler System Requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_.

Intel(R) Extension for Scikit-learn is compatible with the last four versions of scikit-learn:

* 1.0.X
* 1.1.X
* 1.2.X 
* 1.3.X

Memory Requirements
**********************
By default, algorithms in |intelex| run in the multi-thread mode. This mode uses all available threads. 
Optimized scikit-learn algorithms can consume more RAM than their corresponding unoptimized versions.

.. list-table::
   :header-rows: 1
   :align: left

   * - Algorithm
     - Single-thread mode
     - Multi-thread mode
   * - SVM
     - Both Scikit-learn and |intelex| consume approximately the same amount of RAM.
     - In |intelex|, an algorithm with ``N`` threads consumes ``N`` times more RAM.

In all |intelex| algorithms with GPU support, computations run on device memory. 
The device memory must be large enough to store a copy of the entire dataset.
You may also require additional device memory for internal arrays that are used in computation.


.. seealso::

   :ref:`Samples<samples>`
