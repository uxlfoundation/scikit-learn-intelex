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
.. _array_api:

=================
Array API support
=================

Overview
========

|sklearnex| has partial support for input data classes that conform to the
`Array API <https://data-apis.org/array-api/>`_ specification, such as :external+dpnp:doc:`dpnp.ndarray <reference/ndarray>`
or `torch.tensor <https://docs.pytorch.org/docs/stable/tensors.html>`__. This is particularly
useful for GPU computations, as it allows performing operations on inputs that are already
on GPU without moving the data from host to device.

When passing array API inputs whose data is on a SyCL-enabled device (e.g. an Intel GPU), as
supported for example by `PyTorch <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`__
and |dpnp|, computations will be performed on the device where the data lives, without involving any
data transfers. Note that all of the inputs (e.g. ``X`` and ``y`` passed to ``.fit()`` methods) must
be allocated on the same device for this to work.

While other array API inputs (e.g. torch arrays with data allocated on a non-SyCL device) are supported
by the |sklearnex|, note that the data will be transferred to host if it isn't already, and the
computations will happen on CPU. If array API is :external+sklearn:doc:`enabled for scikit-learn <modules/array_api>`,
then array API inputs whose data is allocated neither on CPU nor on a SyCL device will be forwarded
directly to the unpatched methods from |sklearn|, without using the accelerated versions from this library.

When passing array API inputs to methods such as ``.predict()``, the output will be of the same class
as the inputs, but be aware that array attributes of fitted models (e.g. ``coef_`` in a linear model) will not
necessarily be of the same class as array API inputs passed to ``.fit()``.

.. note::
    Unlike |sklearn|, |sklearnex| does not require `array-api-compat <https://github.com/data-apis/array-api-compat>`__
    to be installed for array API support.

.. note::
    As SyCL arrays from array API classes (such as :external+dpnp:doc:`dpnp.ndarray <reference/ndarray>`)
    contain SyCL contexts, they do not require ``config_context(target_offload=device)`` to run on SyCL devices.
    However, if such inputs are used under a ``config_context``, it will override the data's SyCL context
    and might force movement of data to the targeted device.


Supported classes
=================

The following patched classes have support for array API inputs:

- :obj:`sklearnex.basic_statistics.BasicStatistics`
- :obj:`sklearnex.basic_statistics.IncrementalBasicStatistics`
- :obj:`sklearn.cluster.DBSCAN`
- :obj:`sklearn.covariance.EmpiricalCovariance`
- :obj:`sklearnex.covariance.IncrementalEmpiricalCovariance`
- :obj:`sklearn.decomposition.PCA`
- :obj:`sklearn.linear_model.LinearRegression`
- :obj:`sklearn.linear_model.Ridge`
- :obj:`sklearnex.linear_model.IncrementalLinearRegression`
- :obj:`sklearnex.linear_model.IncrementalRidge`

.. note::
    While full array API support is currently not implemented for all classes, :external+dpnp:doc:`dpnp.ndarray <reference/ndarray>`
    and :external+dpctl:doc:`dpctl.tensor <api_reference/dpctl/tensor>` inputs are supported by all the classes
    that have :ref:`GPU support <oneapi_gpu>`.


Example usage
=============

DPNP ndarrays
-------------

Example code showcasing how to use :external+dpnp:doc:`dpnp.ndarray <reference/ndarray>` arrays to
run patched :obj:`sklearn.ensemble.RandomForestRegressor` on a GPU without ``config_context(array_api_dispatch=True)``:

.. toggle::

    .. literalinclude:: ../../examples/sklearnex/random_forest_regressor_dpnp.py
           :language: python

DPCTL usm_ndarrays
------------------
Example code showcasing how to use :external+dpctl:doc:`dpctl.tensor <api_reference/dpctl/tensor>` arrays to run
patched :obj:`sklearn.ensemble.RandomForestClassifier` on a GPU without ``config_context(array_api_dispatch=True)``:

.. toggle::

    .. literalinclude:: ../../examples/sklearnex/random_forest_classifier_dpctl.py
           :language: python

As on previous example, if |dpctl| array API namespace was used for training, then fitted attributes will be on the
CPU, as :obj:`numpy.ndarray` class.

``array-api-strict``
--------------------

Example code showcasing how to use `array-api-strict <https://github.com/data-apis/array-api-strict>`__
arrays to run patched :obj:`sklearn.cluster.DBSCAN`.

.. toggle::

    .. literalinclude:: ../../examples/sklearnex/dbscan_array_api.py
           :language: python
