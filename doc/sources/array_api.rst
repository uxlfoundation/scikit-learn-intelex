.. Copyright 2024 Intel Corporation
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

.. _array_api:

=================
Array API support
=================
The `Array API <https://data-apis.org/array-api/latest/>`_ specification defines
a standard API for all array manipulation libraries with a NumPy-like API.
Intel(R) Extension for Scikit-Learn doesn't require
`array-api-compat <https://github.com/data-apis/array-api-compat>`__ to be installed for
functional support of the array API standard.
In the current implementation, the functional support of array api follows the functional
support of different array or DataFrame inputs and does not modify the precision of the
input and output data formats unless necessary. Any array API input will be converted to host
numpy.ndarrays and all internal manipulations with data will be done with these representations of
the input data. DPNP's 'ndarray' and Data Parallel Control's 'usm_ndarray' have special handling
requirements that are described in the relevant section of this document. Output values will in
all relevant cases match the input data format.

.. note::
    Currently, only `array-api-strict <https://github.com/data-apis/array-api-strict>`__,
    `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__, `dpnp <https://github.com/IntelPython/dpnp>`__
    and `numpy <https://numpy.org/>`__ are known to work with sklearnex estimators.
.. note::
    Stock Scikit-learn’s array API support requires `array-api-compat <https://github.com/data-apis/array-api-compat>`__ to be installed.


Support for DPNP and DPCTL
==========================
The functional support of input data for sklearnex estimators also extended for SYCL USM array types.
These include SYCL USM arrays `dpnp's <https://github.com/IntelPython/dpnp>`__ ndarray and
`Data Parallel Control usm_ndarray <https://intelpython.github.io/dpctl/latest/index.html>`__.
DPNP ndarray and Data Parallel Control usm_ndarray contain SYCL contexts which can be used for
`sklearnex` device offloading.

.. note::
    Current support for DPNP and DPCTL usm_ndarray data can be copied and moved to and from device in sklearnex and have
    impacts on memory utilization.

DPCTL or DPNP inputs are not required to use `config_context(target_offload=device)`.
`sklearnex` will use input usm_ndarray sycl context for device offloading.

.. note::
    As DPCTL or DPNP inputs contain SYCL contexts, they do not require `config_context(target_offload=device)`.
    However, the use of `config_context`` will override the contained SYCL context and will force movement
    of data to the targeted device.


Support for Array API-compatible inputs
=======================================
All patched estimators, metrics, tools and non-scikit-learn estimators functionally support Array API standard.
Intel(R) Extension for scikit-Learn preserves input data format for all outputs. For all array inputs except
SYCL USM arrays `dpnp's <https://github.com/IntelPython/dpnp>`__ ndarray and
`Data Parallel Control usm_ndarray <https://intelpython.github.io/dpctl/latest/index.html>`__ all computation
will be only accomplished on CPU unless specified by a `config_context`` with an available GPU device.

Stock scikit-learn uses `config_context(array_api_dispatch=True)` for enabling Array API
`support <https://scikit-learn.org/1.5/modules/array_api.html>`__.
If `array_api_dispatch` is enabled and the installed Scikit-Learn version supports array API, then the original
inputs are used when falling back to Scikit-Learn functionality.

.. note::
    Data Parallel Control usm_ndarray or DPNP ndarray inputs will use host numpy data copies when
    falling back to Scikit-Learn since they are not array API compliant.
.. note::
    Functional support doesn't guarantee that after the model is trained, fitted attributes that are arrays
    will also be from the same namespace as the training data.


Example usage
=============

DPNP ndarrays
-------------

Here is an example code to demonstrate how to use `dpnp <https://github.com/IntelPython/dpnp>`__ arrays to
run `RandomForestRegressor` on a GPU without `config_context(array_api_dispatch=True)`:

.. literalinclude:: ../../examples/sklearnex/random_forest_regressor_dpnp.py
	   :language: python


.. note::
    Functional support doesn't guarantee that after the model is trained, fitted attributes that are arrays
    will also be from the same namespace as the training data.

For example, if `dpnp's <https://github.com/IntelPython/dpnp>`__ namespace was used for training,
then fitted attributes will be on the CPU and `numpy.ndarray` data format.

DPCTL usm_ndarrays
------------------
Here is an example code to demonstrate how to use `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__
arrays to run `RandomForestClassifier` on a GPU witout `config_context(array_api_dispatch=True)`:

.. literalinclude:: ../../examples/sklearnex/random_forest_classifier_dpctl.py
	   :language: python

As on previous example, if `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__ Array API namespace was
used for training, then fitted attributes will be on the CPU and `numpy.ndarray` data format.

Use of `array-api-strict`
-------------------------

Here is an example code to demonstrate how to use `array-api-strict <https://github.com/data-apis/array-api-strict>`__
arrays to run `DBSCAN`.

.. literalinclude:: ../../examples/sklearnex/dbscan_array_api.py
	   :language: python
