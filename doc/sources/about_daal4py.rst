.. Copyright contributors to the oneDAL project
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

.. _about_daal4py:

About daal4py
=============

Introduction
------------

``daal4py`` is a low-level module within the |sklearnex| package providing Python bindings
over the |onedal|. It has been deprecated in favor of the newer ``sklearnex`` module in the
same package, which offers a more idiomatic and higher-level interface for calling accelerated
routines from the |onedal| in Python.

Internally, ``daal4py`` is a Python wrapper over the `now-deprecated "DAAL" interface <https://uxlfoundation.github.io/oneDAL/index.html#oneapi-vs-daal>`__
of the |onedal|, while ``sklearnex`` is a module built atop of the "oneAPI" interface, offering
DPC-based features such as :ref:`GPU support <oneapi_gpu>`.

There is a large degree of overlap in the functionalities offered between the two modules
``daal4py`` and ``sklearnex`` - module ``sklearnex`` should be prefered whenever possible,
either by using it directly or through the :ref:`patching mechanism <patching>` - but ``daal4py``
exposes some additional functionalities from the |onedal| that ``sklearnex`` doesn't:

- :ref:`Algorithms that are outside the scope of scikit-learn <non_sklearn_d4p>`.
- :ref:`Distributed mode on CPU <distributed_daal4py>`.
- Fast serving of gradient boosted decision trees from other libraries such as XGBoost
  (:ref:`model builders <model_builders>`).

Previously ``daal4py`` was distributed as a separate package, but it is now an importable module
within the ``scikit-learn-intelex`` package - meaning, after installing ``scikit-learn-intelex``,
it can be imported as follows:

.. code::

    import daal4py

For documentation about specific functions, see the :ref:`daal4py API reference <daal4py_ref>`.


Using daal4py
-------------

Unlike ``sklearnex``, ``daal4py``, being a lower-level interface, does not follow scikit-learn
idioms - instead, the process for calling procedures from the ``daal4py`` interface is as follows:

- Instantiate an 'algorithm' class by calling its contructor, without any data - for example:
  ``qr_algo = daal4py.qr()``.
- Call the 'compute' method of that instantiated algorithm in order to obtain a 'result' object,
  passing it the data on which it will operate - for example: ``qr_result = qr_algo.compute(X)``.
- Access the relevant results in the 'result' object - for example: ``R = qr_result.matrixR``.


Full example calling the QR algorithm:

.. code::

    import daal4py
    import numpy as np

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(100,5))

    qr_algo = daal4py.qr()
    qr_result = qr_algo.compute(X)

    np.testing.assert_almost_equal(
        np.abs(  qr_result.matrixR  ),
        np.abs(  np.linalg.qr(X).R  ),
    )

.. note::
    QR factorization, unlike other linear algebra procedures, does not have a strictly unique
    solution - if the signs (+/-) of numbers are flipped for a particular column in both the Q
    and R matrices, they would still be valid and equivalent QR factorizations of the same
    original matrix 'X'.

    Procedures like Cholesky decomposition are typically constrained to have only positive signs
    in the main diagonal in order to make the results deterministic, but this is not always the
    case for QR in most software, hence the example above takes the absolute values when comparing
    results from different libraries.


Streaming mode
**************

Many algorithms in ``daal4py`` accept an argument ``streaming=True``, which allows executing the
computations in a 'streaming' or 'online' fashion, by supplying it different subsets of the data,
one at a time (batches), instead of passing the whole data upfront, while still arriving at the
same final result as if all the data had been passed at once.

.. note::
    The ``sklearnex`` module also offers incremental versions of some algorithms - see the docs
    on :ref:`extension_estimators` for more details.

This can be useful for executing algorithms on large datasets that don't fit in memory but which
can still be loaded in smaller chunks, or for machine learning models that are constantly being
updated as new data is collected, for example.

In order to use streaming mode, the algorithm constructor needs to be passed argument ``streaming=True``,
method ``.compute()`` needs to be called multiple times with different data, and the 'result'
object should be obtained by calling method ``.finalize()`` after all the data has been passed.

Example: ::

    import daal4py
    import numpy as np

    rng = np.random.default_rng(seed=123)
    X_full = rng.standard_normal(size=(100,5))
    batches = np.split(np.arange(100), 5)

    qr_algo = daal4py.qr(streaming=True)
    for batch in batches:
        X_batch = X_full[batch]
        qr_algo.compute(X_batch)

    qr_result = qr_algo.finalize()

    np.testing.assert_almost_equal(
        np.abs(  qr_result.matrixR  ),
        np.abs(  np.linalg.qr(X).R  ),
    )

List of algorithms in ``daal4py`` supporting streaming mode:

- :obj:`SVD <daal4py.svd>`
- :obj:`Linear Regression <daal4py.linear_regression_training>`
- :obj:`Ridge Regression <daal4py.ridge_regression_training>`
- :obj:`Multinomial Naive Bayes <daal4py.multinomial_naive_bayes_training>`
- :obj:`Moments of Low Order <daal4py.low_order_moments>`
- :obj:`Covariance <daal4py.covariance>`
- :obj:`QR decomposition <daal4py.qr>`

Distributed mode
****************

Many algorithms in ``daal4py`` accept an argument ``distributed=True``, which allows
running computations in a distributed compute nodes using the MPI framework.

See the section :ref:`distributed_daal4py` for more details.

Documentation
*************

See :ref:`daal4py_ref` for the full documentation of functions and classes.
