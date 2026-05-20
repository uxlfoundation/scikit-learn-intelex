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
.. include:: substitutions.rst
.. _input_types:

=====================
Supported input types
=====================

Just like |sklearn| estimators, estimators from the |sklearnex| are able to work with different classes of input data, including:

- :external+numpy:doc:`Numpy arrays <user/whatisnumpy>`.

  - Note: :external+numpy:doc:`masked arrays <reference/maskedarray>` are also supported, but just like in stock |sklearn|, the
    underlying array values are used without the mask.
- Other array classes implementing the `Array API <https://data-apis.org/array-api/latest/>`__ protocol
  (see :ref:`array_api` for details).
- SciPy :external+scipy:doc:`sparse arrays and sparse matrices <tutorial/sparse>` (depending on the estimator - see :doc:`algorithms`).
- Pandas :external+pandas:doc:`DataFrame and Series <user_guide/dsintro>` classes.
- Other DataFrame classes recognize by data validators from |sklearn|, such as the ones from `Polars <https://pola.rs>`__.

In addition, |sklearnex| also supports |dpnp_array| arrays (with and without array API mode) in estimators with
:ref:`GPU support <sklearn_algorithms_gpu>` (see also :doc:`array_api`).

|sklearnex| currently does not offer accelerated routines for input types not listed here - when
receiving an unsupported class, estimators will either convert to a supported class under some
circumstances (e.g. PyArrow tables might get converted to NumPy arrays when passed to data
validators from stock |sklearn|), throw an error (e.g. when passing some data format not that's
not recognized by |sklearn|), or fall back to stock |sklearn| to handle it (when array API is
enabled but the input is unsupported).

.. warning::
  In some cases, data passed to estimators might be copied/duplicated during calls to methods such as fit/predict.
  The affected cases are listed below:

  - Non-contiguous NumPy array - i.e. where strides are wider than one element across both rows and columns.
  - For SciPy CSR matrix / array, index arrays are always copied. Note that sparse matrices in formats other than CSR
    will be converted to CSR, which implies more than just data copying.
  - Heterogeneous NumPy array.
  - If a SYCL queue is used for :ref:`target_offload` for a device without ``float64`` support but data is ``float64``,
    data will be converted to ``float32``.
  - If a |dpnp_array| array on GPU is used as input without :doc:`array_api` being enabled, then data will be transferred
    to CPU for validations and then back to GPU for computations (see :doc:`oneapi-gpu` for details).
  - If a |dpnp_array| array on GPU is passed to an estimator with :ref:`GPU support <sklearn_algorithms_gpu>` but the
    requested operation is not supported on GPU, and if array API is not enabled or |sklearn| does not support array API
    for the requested operation, then data might be transferred to CPU and the operation done there (see :doc:`config-contexts`).
