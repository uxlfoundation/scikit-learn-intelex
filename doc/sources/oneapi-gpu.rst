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

.. include:: substitutions.rst
.. _oneapi_gpu:

##############################################################
oneAPI and GPU support in |sklearnex|
##############################################################

|sklearnex| can execute computations on different devices (CPUs and GPUs, including integrated GPUs from laptops and desktops) through the SYCL framework in oneAPI.

The device used for computations can be easily controlled through the target offloading functionality (e.g. through ``sklearnex.config_context(target_offload="gpu")``, which moves data to GPU if it's not already there - see rest of this page for more details), but for finer-grained controlled (e.g. operating on arrays that are already in a given device's memory), it can also interact with objects from package |dpctl|, which offers a Python interface over SYCL concepts such as devices, queues, and USM (unified shared memory) arrays.

While not strictly required, package |dpctl| is recommended for a better experience on GPUs - for example, it can provide GPU-allocated arrays that enable compute-follows-data execution models (i.e. so that ``target_offload`` wouldn't need to move the data from CPU to GPU).

.. important:: Be aware that GPU usage requires non-Python dependencies on your system, such as the `Intel(R) Compute Runtime <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_ (see below).

Prerequisites
-------------

For execution on GPUs, DPC++ runtime and Intel Compute Runtime (also referred to elsewhere as 'GPGPU drivers') are required.

DPC++ Runtime
~~~~~~~~~~~~~

DPC++ compiler runtime can be installed either from PyPI or Conda:

- Install from PyPI::

     pip install dpcpp-cpp-rt

- Install using Conda from Intel's repository::

     conda install -c https://software.repos.intel.com/python/conda/ dpcpp_cpp_rt

- Install using Conda from the conda-forge channel::

     conda install -c conda-forge dpcpp_cpp_rt

Intel Compute Runtime
~~~~~~~~~~~~~~~~~~~~~

On Windows, GPU drivers for iGPUs and dGPUs include the required Intel Compute Runtime. Drivers for windows can be downloaded from `this link <https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html>`__.

For datacenters, see further instructions `here <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-dpcpp/2025.html#inpage-nav-2-1-1>`__.


On Linux, some distributions - namely Ubuntu Desktop 25.04 and higher, and Fedora Workstation 42 and higher - come with the compute runtime for iGPUs and dGPUs preinstalled, while others require installing them separately.

Debian systems require installing package ``intel-opencl-icd`` (along with its dependencies such as ``intel-compute-runtime`` and ``intel-graphics-compiler``), which is available from Debian's ``main`` repository: ::

    sudo apt-get install intel-opencl-icd

.. tip:: For Debian Trixie (13), the Intel Compute Runtime is not available from the Stable repository, but can be installed by enabling the Sid (Unstable) repository.

For Arch Linux, and for other distributions in general, see the `GPGPU article in the Arch wiki <https://wiki.archlinux.org/title/GPGPU>`__.

.. important::
    If using the |sklearnex| in a conda environment, GPU support requires the the OpenCL ICD package `for conda <https://github.com/IntelPython/intel-gpu-ocl-icd-system-feedstock>`__ to be installed in the conda environment, **in addition to the system install** of the same package: ::

        conda install -c https://software.repos.intel.com/python/conda/ intel-gpu-ocl-icd-system

Be aware that datacenter-grade devices, such as 'Flex' and 'Max', require different drivers and runtimes. For CentOS and for datacenter-grade devices, see `instructions here <https://dgpu-docs.intel.com/driver/installation.html>`__.

For more details, see the `DPC++ requirements page <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-dpcpp/2025.html>`__.

Device offloading
-----------------

|sklearnex| offers two options for running an algorithm on a specified device:

- Use global configurations of |sklearnex|\*:

  1. The :code:`target_offload` argument (in ``config_context`` and in ``set_config`` / ``get_config``)
     can be used to set the device primarily used to perform computations. Accepted data types are
     :code:`str` and :obj:`dpctl.SyclQueue`. Strings must match to device names recognized by
     the SYCL* device filter selector - for example, ``"gpu"``. If passing ``"auto"``,
     the device will be deduced from the location of the input data. Examples:

     .. code-block:: python
        
        from sklearnex import config_context
        from sklearnex.linear_model import LinearRegression
        
        with config_context(target_offload="gpu"):
            model = LinearRegression().fit(X, y)

     .. code-block:: python
        
        from sklearnex import set_config
        from sklearnex.linear_model import LinearRegression
        
        set_config(target_offload="gpu")
        model = LinearRegression().fit(X, y)


     If passing a string different than ``"auto"``,
     it must be a device 

  2. The :code:`allow_fallback_to_host` argument in those same configuration functions
     is a Boolean flag. If set to :code:`True`, the computation is allowed
     to fallback to the host device when a particular estimator does not support
     the selected device. The default value is :code:`False`.

These options can be set using :code:`sklearnex.set_config()` function or
:code:`sklearnex.config_context`. To obtain the current values of these options,
call :code:`sklearnex.get_config()`.

.. note::
     Functions :code:`set_config`, :code:`get_config` and :code:`config_context`
     are always patched after the :code:`sklearnex.patch_sklearn()` call.

- Pass input data as :obj:`dpctl.tensor.usm_ndarray` to the algorithm.

  The computation will run on the device where the input data is
  located, and the result will be returned as :code:`usm_ndarray` to the same
  device.

  .. important::
    In order to enable zero-copy operations on GPU arrays, it's necessary to enable
    :ref:`array API support <array_api>` for scikit-learn. Otherwise, if passing a GPU
    array and array API support is not enabled, GPU arrays will first be transferred to
    host and then back to GPU.

  .. note::
    All the input data for an algorithm must reside on the same device.

  .. warning::
    The :code:`usm_ndarray` can only be consumed by the base methods
    like :code:`fit`, :code:`predict`, and :code:`transform`.
    Note that only the algorithms in |sklearnex| support
    :code:`usm_ndarray`. The algorithms from the stock version of |sklearn|
    do not support this feature.


Example
-------

A full example of how to patch your code with Intel CPU/GPU optimizations:

.. code-block:: python

   from sklearnex import patch_sklearn, config_context
   patch_sklearn()

   from sklearn.cluster import DBSCAN

   X = np.array([[1., 2.], [2., 2.], [2., 3.],
                 [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
   with config_context(target_offload="gpu:0"):
       clustering = DBSCAN(eps=3, min_samples=2).fit(X)


.. note:: Current offloading behavior restricts fitting and predictions (a.k.a. inference) of any models to be
     in the same context or absence of context. For example, a model whose ``.fit()`` method was called in a GPU context with
     ``target_offload="gpu:0"`` will throw an error if a ``.predict()`` call is then made outside the same GPU context.
