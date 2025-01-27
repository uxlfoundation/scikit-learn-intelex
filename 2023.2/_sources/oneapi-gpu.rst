.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. _oneapi_gpu:

##############################################################
oneAPI and GPU support in |intelex|
##############################################################

|intelex| supports oneAPI concepts, which
means that algorithms can be executed on different devices: CPUs and GPUs.
This is done via integration with
`dpctl <https://intelpython.github.io/dpctl/latest/index.html>`_ package that
implements core oneAPI concepts like queues and devices.

Prerequisites
-------------

For execution on GPU, DPC++ compiler runtime and driver are required. Refer to `DPC++ system
requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_ for details.

DPC++ compiler runtime can be installed either from PyPI or Anaconda:

- Install from PyPI::

     pip install dpcpp-cpp-rt

- Install from Anaconda::

     conda install dpcpp_cpp_rt -c intel

Device offloading
-----------------

|intelex| offers two options for running an algorithm on a
specific device with the help of dpctl:

- Pass input data as `dpctl.tensor.usm_ndarray <https://intelpython.github.io/dpctl/latest/docfiles/dpctl/usm_ndarray.html#dpctl.tensor.usm_ndarray>`_ to the algorithm.

  The computation will run on the device where the input data is
  located, and the result will be returned as :code:`usm_ndarray` to the same
  device.

  .. note::
    All the input data for an algorithm must reside on the same device.

  .. warning::
    The :code:`usm_ndarray` can only be consumed by the base methods
    like :code:`fit`, :code:`predict`, and :code:`transform`.
    Note that only the algorithms in |intelex| support
    :code:`usm_ndarray`. The algorithms from the stock version of scikit-learn
    do not support this feature.
- Use global configurations of |intelex|\*:
  
  1. The :code:`target_offload` option can be used to set the device primarily
     used to perform computations. Accepted data types are :code:`str` and
     :code:`dpctl.SyclQueue`. If you pass a string to :code:`target_offload`,
     it should either be ``"auto"``, which means that the execution
     context is deduced from the location of input data, or a string
     with SYCL* filter selector. The default value is ``"auto"``.
  
  2. The :code:`allow_fallback_to_host` option
     is a Boolean flag. If set to :code:`True`, the computation is allowed 
     to fallback to the host device when a particular estimator does not support
     the selected device. The default value is :code:`False`.

These options can be set using :code:`sklearnex.set_config()` function or
:code:`sklearnex.config_context`. To obtain the current values of these options,
call :code:`sklearnex.get_config()`.

.. note::
     Functions :code:`set_config`, :code:`get_config` and :code:`config_context`
     are always patched after the :code:`sklearnex.patch_sklearn()` call.

.. rubric:: Compatibility considerations

For compatibility reasons, algorithms in |intelex| may be offloaded to the device using
:code:`daal4py.oneapi.sycl_context`. However, it is recommended to use one of the options
described above for device offloading instead of using :code:`sycl_context`.

Example
-------

An example on how to patch your code with Intel CPU/GPU optimizations:

.. code-block:: python

   from sklearnex import patch_sklearn, config_context
   patch_sklearn()

   from sklearn.cluster import DBSCAN

   X = np.array([[1., 2.], [2., 2.], [2., 3.],
               [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
   with config_context(target_offload="gpu:0"):
      clustering = DBSCAN(eps=3, min_samples=2).fit(X)

Note: Current offloading behavior restricts fitting and inference of any models to be
in the same context or absence of context. For example, a model trained in the GPU context with
target_offload="gpu:0" throws an error if the inference is made outside the same GPU context.
