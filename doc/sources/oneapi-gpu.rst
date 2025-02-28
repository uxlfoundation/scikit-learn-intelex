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
oneAPI and GPU support in |intelex|
##############################################################

|intelex| can execute computations on different devices (CPUs, GPUs) through the SYCL framework in oneAPI.

The device used for computations can be easily controlled through the target offloading functionality (e.g. through ``sklearnex.config_context(target_offload="gpu")`` - see rest of this page for more details), but for finer-grained controlled (e.g. operating on arrays that are already in a given device's memory), it can also interact with objects from package |dpctl|, which offers a Python interface over SYCL concepts such as devices, queues, and USM (unified shared memory) arrays.

While not strictly required, package |dpctl| is recommended for a better experience on GPUs.

.. important:: Be aware that GPU usage requires non-Python dependencies on your system, such as the `Intel(R) GPGPU Drivers <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_.

Prerequisites
-------------

For execution on GPUs, DPC++ runtime and GPGPU drivers are required.

DPC++ compiler runtime can be installed either from PyPI or Conda:

- Install from PyPI::

     pip install dpcpp-cpp-rt

- Install using Conda from Intel's repository::

     conda install -c https://software.repos.intel.com/python/conda/ dpcpp_cpp_rt

- Install using Conda from the conda-forge channel::

     conda install -c conda-forge dpcpp_cpp_rt

For GPGPU driver installation instructions, see the general `DPC++ system requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_ sections corresponding to your operating system.

Device offloading
-----------------

|intelex| offers two options for running an algorithm on a specified device:

- Use global configurations of |intelex|\*:

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

  .. note::
    All the input data for an algorithm must reside on the same device.

  .. warning::
    The :code:`usm_ndarray` can only be consumed by the base methods
    like :code:`fit`, :code:`predict`, and :code:`transform`.
    Note that only the algorithms in |intelex| support
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
