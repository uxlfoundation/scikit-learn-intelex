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

Many estimators from the |sklearnex| support passing data classes that conform to the
`Array API <https://data-apis.org/array-api/>`_ specification as inputs to methods like ``.fit()``
and ``.predict()``, such as :external+dpnp:doc:`dpnp.ndarray <reference/ndarray>` or
`torch.tensor <https://docs.pytorch.org/docs/stable/tensors.html>`__. This is particularly
useful for GPU computations, as it allows performing operations on inputs that are already
on GPU without moving the data from host to device.

.. important::
    Array API is disabled by default in |sklearn|. In order to get array API support in the |sklearnex|, it must
    be :external+sklearn:doc:`enabled in scikit-learn <modules/array_api>`, which requires either changing
    global settings or using a ``config_context``.

When passing array API inputs whose data is on a SyCL-enabled device (e.g. an Intel GPU), as
supported for example by `PyTorch <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`__
and |dpnp|, if array API support is enabled and the requested operation (e.g. call to ``.fit()`` / ``.predict()``
on the estimator class being used) is :ref:`supported on device/GPU <sklearn_algorithms_gpu>`, computations
will be performed on the device where the data lives, without involving any data transfers. Note that all of
the inputs (e.g. ``X`` and ``y`` passed to ``.fit()`` methods) must be allocated on the same device for this to
work. If the requested operation is not supported on the device where the data lives, then it will either fall
back to |sklearn|, or to an accelerated CPU version from the |sklearnex| when supported - these are controllable
through options ``allow_sklearn_after_onedal`` (default is ``True``) and ``allow_fallback_to_host`` (default is
``False``), respectively, which are accepted by ``config_context`` and ``set_config`` after
:ref:`patching scikit-learn <patching>` or when importing those directly from ``sklearnex``.

.. note::
    Under default settings for ``set_config`` / ``config_context``, operations that are not supported on GPU will
    fall back to |sklearn| instead of falling back to CPU versions from the |sklearnex|.

If array API is enabled for |sklearn| and the estimator being used has array API support on |sklearn| (which can be
verified by attribute ``array_api_support`` from :obj:`sklearn.utils.get_tags`), then array API inputs whose data
is allocated neither on CPU nor on a SyCL device will be forwarded directly to the unpatched methods from |sklearn|,
without using the accelerated versions from this library, regardless of option ``allow_sklearn_after_onedal``.

While other array API inputs (e.g. torch arrays with data allocated on a non-SyCL device) might be supported
by the |sklearnex| in cases where the same class from |sklearn| doesn't support array API, note that the data will
be transferred to host if it isn't already, and the computations will happen on CPU.

.. hint::
    Enable :ref:`verbose` to see information about whether data transfers happen during an operation or not,
    whether an accelerated version from the extension is used, and where (CPU/device) the operation is executed.

When passing array API inputs to methods such as ``.predict()`` of estimators with array API support, the output
will always be of the same class as the inputs, but be aware that array attributes of fitted models (e.g. ``coef_``
in a linear model) will not necessarily be of the same class as array API inputs passed to ``.fit()``, even though
in many cases they are.

.. warning::
    If array API inputs are passed to an estimator's ``.fit()``, subsequent data passed to methods such as
    ``.predict()`` or ``.score()`` of the fitted model might be of a different class than the ``X``/``y`` passed to
    ``.fit()``, but **it must reside on the same device** - meaning: a model that was fitted with GPU arrays cannot
    make predictions on CPU arrays, and a model fitted with CPU array API inputs cannot make predictions on GPU
    arrays, even if they are of the same class. Attempting to pass data on the wrong device might lead to
    process-wide crashes.

.. note::
    Unlike |sklearn|, |sklearnex| does not require `array-api-compat <https://github.com/data-apis/array-api-compat>`__
    to be installed for array API support.

.. note::
    The ``target_offload`` option in config contexts and settings is not intended to work with array API
    classes that have :external+dpctl:doc:`USM data <dpctl/memory>`. In order to ensure that computations
    happen on the intended device under array API, make sure that the data is already on the desired device.


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
    that have :ref:`GPU support <oneapi_gpu>`. Note however that if array API support is not enabled in |sklearn|,
    when passing these classes as inputs, data will be transferred to host and then back to device instead of being
    used directly.


Example usage
=============

GPU operations on GPU arrays
----------------------------

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

    # Fitted attributes are now of the same class as inputs
    assert isinstance(model.coef_, X.__class__)

    # Predictions are also of the same class
    with config_context(array_api_dispatch=True):
        pred = model.predict(X[:5])
    assert isinstance(pred, X.__class__)

    # Fitted models can be passed array API inputs of a different class
    # than the training data, as long as their data resides in the same
    # device. This now fits a model using a non-NumPy class whose data is on CPU.
    X_cpu = dpnp.array(X_np, device="cpu")
    y_cpu = dpnp.array(y_np, device="cpu")
    model_cpu = LinearRegression()
    with config_context(array_api_dispatch=True):
        model_cpu.fit(X_cpu, y_cpu)
        pred_dpnp = model_cpu.predict(X_cpu[:5])
        pred_np = model_cpu.predict(X_cpu[:5].asnumpy())
    assert isinstance(pred_dpnp, X_cpu.__class__)
    assert isinstance(pred_np, np.ndarray)
    assert pred_dpnp.__class__ != pred_np.__class__


``array-api-strict``
--------------------

Example code showcasing how to use `array-api-strict <https://github.com/data-apis/array-api-strict>`__
arrays to run patched :obj:`sklearn.cluster.DBSCAN`.

.. toggle::

    .. literalinclude:: ../../examples/sklearnex/dbscan_array_api.py
           :language: python
