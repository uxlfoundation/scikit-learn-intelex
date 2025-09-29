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

.. _input_types:

=====================
Supported input types
=====================

Just like Scikit-Learn estimators, estimators from Extension are able to accept
and work with different classes of input data, including:

- `Numpy <https://numpy.org/>`__ arrays.
- Other array classes implementing the `Array API <https://data-apis.org/array-api/latest/>`__ protocol
  (see :ref:`array_api` for details).
- `SciPy sparse arrays and sparse matrices <https://docs.scipy.org/doc/scipy/tutorial/sparse.html>`__ (depending on the estimator).
- `Pandas <https://pandas.pydata.org>`__ ``DataFrame`` and ``Series`` classes.

In addition, Extension also supports:

- `dpnp <https://github.com/IntelPython/dpnp>`__ arrays.
- `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`__ arrays.

Stock Scikit-Learn estimators, depending on the version, might offer support for additional
input types beyond this list, such as ``DataFrame`` and ``Series`` classes from other libraries
like `Polars <https://pola.rs>`__.

Extension currently does not offer accelerated routines for input types not listed
here - when receiving an unsupported class, estimators will fall back to stock Scikit-Learn to
handle it, so make sure to convert them to a supported type when using Extension.

.. warning::
  In certain cases data passed to estimators might be copied/duplicated during calls to methods such as fit/predict under some circumstances.
  The affected cases are listed below.

  - Non-contiguous NumPy array - i.e. where strides are wider than one element across both rows and columns
  - For SciPy CSR matrix / CSR array index array is always copied.
  - Heterogeneous NumPy array
  - If :ref:`Array API <array_api>` is not enabled then data from GPU devices are always copied to the host device and then result table 
    (for applicable methods) is copied to the source device.
