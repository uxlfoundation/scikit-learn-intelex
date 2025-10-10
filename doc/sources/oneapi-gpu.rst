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

###########
GPU support
###########

Overview
--------

|sklearnex| can execute computations on different devices (CPUs and GPUs, including integrated GPUs from laptops and desktops) supported by the SyCL framework.

The device used for computations can be easily controlled through the ``target_offload`` option in config contexts, which moves data to GPU if it's not already there - see :ref:`config_contexts` and rest of this page for more details).

For finer-grained controlled (e.g. operating on arrays that are already in a given device's memory), it can also interact with on-device :ref:`array API classes <array_api>` like |dpnp_array|, and with SyCL-related objects from package |dpctl| such as :obj:`dpctl.SyclQueue`.

.. Note:: Note that not every operation from every estimator is supported on GPU - see the :ref:`GPU support table <sklearn_algorithms_gpu>` for more information.

.. important:: Be aware that GPU usage requires non-Python dependencies on your system, such as the `Intel(R) Compute Runtime <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_ (see below).

Software Requirements
---------------------

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

Running operations on GPU
-------------------------

|sklearnex| offers different options for running an algorithm on a specified device (e.g. a GPU):

Target offload option
~~~~~~~~~~~~~~~~~~~~~

Just like |sklearn|, the |sklearnex| can use configuration contexts and global options to modify how it interacts with different inputs - see :ref:`config_contexts` for details.

In particular, the |sklearnex| allows an option ``target_offload`` which can be passed a SyCL device name like ``"gpu"`` indicating where the operations should be performed, moving the data to that device in the process if it's not already there; or a :obj:`dpctl.SyclQueue` object from an already-existing queue on a device.

Example:

.. tabs::
    .. tab:: Passing a device name
       .. code-block:: python

           from sklearnex import config_context
           from sklearnex.linear_model import LinearRegression
           from sklearn.datasets import make_regression
           X, y = make_regression()
           model = LinearRegression()

           with config_context(target_offload="gpu"):
               model.fit(X, y)
               pred = model.predict(X)

    .. tab:: Passing a SyCL queue
       .. code-block:: python

           import dpctl
           from sklearnex import config_context
           from sklearnex.linear_model import LinearRegression
           from sklearn.datasets import make_regression
           X, y = make_regression()
           model = LinearRegression()

           queue = dpctl.SyclQueue("gpu")
           with config_context(target_offload=queue):
               model.fit(X, y)
               pred = model.predict(X)


.. warning::
    When using ``target_offload``, operations on a fitted model must be executed under a context or global option with the same device or queue where the model was fitted - meaning: a model fitted on GPU cannot make predictions on CPU, and vice-versa. Note that upon serialization and subsequent deserialization of models, data is moved to the CPU.

GPU arrays through array API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As another option, computations can also be performed on data that is already on a SyCL device without moving it there if it belongs to an array API-compatible class, such as |dpnp_array| or `torch.tensor <https://docs.pytorch.org/docs/stable/tensors.html>`__.

This is particularly useful when multiple operations are performed on the same data (e.g. cross validators, stacked ensembles, etc.), or when the data is meant to interact with other libraries besides the |sklearnex|. Be aware that it requires enabling array API support in |sklearn|, which comes with additional dependencies.

See :ref:`array_api` for details, instructions, and limitations. Example:

.. code-block:: python

    # Array API support from sklearn requires enabling it on SciPy too
    import os
    os.environ["SCIPY_ARRAY_API"] = "1"

    import numpy as np
    import dpnp
    from sklearnex import config_context
    from sklearnex.linear_model import LinearRegression

    # Random data for a regression problem
    rng = np.random.default_rng(seed=123)
    X_np = rng.standard_normal(size=(100, 10), dtype=np.float32)
    y_np = rng.standard_normal(size=100, dtype=np.float32)

    # DPNP offers an array-API-compliant class where data can be on GPU
    X = dpnp.array(X_np, device="gpu")
    y = dpnp.array(y_np, device="gpu")

    # Important to note again that array API must be enabled on scikit-learn
    model = LinearRegression()
    with config_context(array_api_dispatch=True):
        model.fit(X, y)

.. note::
    Not all estimator classes in the |sklearnex| support array API objects - see the list of :ref:`estimators with array API support <array_api_estimators>` for details.

DPNP Arrays
~~~~~~~~~~~

As a special case, GPU arrays from |dpnp| can be used without enabling array API, even for estimators in the |sklearnex| that do not currently support array API, but note that it involves data movement to host and back and is thus not the most efficient route in computational terms.

Example:

.. code-block:: python

    import numpy as np
    import dpnp
    from sklearnex import config_context
    from sklearnex.linear_model import LinearRegression

    rng = np.random.default_rng(seed=123)
    X_np = rng.standard_normal(size=(100, 10), dtype=np.float32)
    y_np = rng.standard_normal(size=100, dtype=np.float32)

    X = dpnp.array(X_np, device="gpu")
    y = dpnp.array(y_np, device="gpu")

    model = LinearRegression()
    model.fit(X, y)


Note that, if array API had been enabled, the snippet above would use the data as-is on the device where it resides, but without array API, it implies data movements using the SyCL queue contained by those objects.

.. note::
    All the input data for an algorithm must reside on the same device.
