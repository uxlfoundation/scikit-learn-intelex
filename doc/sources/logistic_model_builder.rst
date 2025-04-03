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

.. _logistic_model_builder:

Serving Logistic Regression from Other Libraries
================================================

The |sklearnex| offers a prediction-only class :obj:`daal4py.mb.LogisticDAALModel` which
can be used to accelerate calculations of predictions from logistic regression models
(both binary and multinomial) produced by other libraries such as |sklearn|.

Logistic regression models from |sklearn| (classes :obj:`sklearn.linear_model.LogisticRegression`
and :obj:`sklearn.linear_model.SGDClassifier`) can be converted to daal4py's
:obj:`daal4py.mb.LogisticDAALModel` through function :obj:`daal4py.mb.convert_model` in the
model builders (``mb``) submodule, which can also convert gradient-boosted decision tree models
(:ref:`model_builders`). See :ref:`about_daal4py` for more information about the
``daal4py`` module.

Example
-------

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    import numpy as np
    import daal4py

    X, y = load_iris(return_X_y=True)
    model_skl = LogisticRegression().fit(X, y)
    model_d4p = daal4py.mb.convert_model(model_skl)

    np.testing.assert_almost_equal(
        model_d4p.predict(X),
        model_skl.predict(X),
    )

    np.testing.assert_almost_equal(
        model_d4p.predict_proba(X),
        model_skl.predict_proba(X),
    )

Details
-------

Acceleration is achieved by leveraging the `MKL library <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__
for both faster linear algebra operations and faster exponentials / logarithms on Intel hardware,
and optionally by using a lower-precision data type (``np.float32``) than what :obj:`sklearn.linear_model.LogisticRegression`
uses.

If you are already using Python libraries (NumPy and SciPy) linked against MKL - for example, by
installing them from the Intel conda channel ``https://software.repos.intel.com/python/conda/`` - then
this class might not offer much of a speedup over |sklearn|, but otherwise it offers an easy way to
speed up inference by better utilizing capabilities of Intel hardware.

Note that besides the :obj:`daal4py.mb.convert_model` function, class :obj:`daal4py.mb.LogisticDAALModel`
can also be instantiated directly from arrays of fitted coefficients and intercepts, thereby allowing to
create predictors out of models from other libraries beyond |sklearn|, such as `glum <https://glum.readthedocs.io/>`__.

Documentation
-------------

See the section about :ref:`model builders <model_builders_docs>` in the ``daal4py`` API reference
for full documentation.
