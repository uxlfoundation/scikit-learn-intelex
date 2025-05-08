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

.. _model_builders:

Serving GBT models from other libraries
=======================================

Introduction
------------

The |sklearnex| can be used to accelerate computation of predictions (also known as "inference")
from GBT (gradient-boosted decision trees) models produced by other libraries such as XGBoost, by
converting those models to class :obj:`daal4py.mb.GBTDAALModel` which offers faster methods
:meth:`daal4py.mb.GBTDAALModel.predict` and :meth:`daal4py.mb.GBTDAALModel.predict_proba`.

.. figure:: model_builders_speedup.webp
    :align: center

    Example speedups in predictions compared to prediction method in XGBoost
    on Intel Xeon Platinum 8275CL (see `blog post <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>`__).

Model objects from other GBT libraries can be converted to this daal4py class through function
:obj:`daal4py.mb.convert_model` in the model builders (``mb``) submodule, which can also convert
logistic regression models (:ref:`logistic_model_builder`). See :ref:`about_daal4py`
for more information about the ``daal4py`` module.

Currently, model builders from ``daal4py`` can work with GBT model objects produced by
the following libraries, both in their base booster classes and their scikit-learn-compatible classes:

- `XGBoost <https://xgboost.readthedocs.io>`__
- `LightGBM <https://lightgbm.readthedocs.io>`__
- `CatBoost <https://catboost.ai/>`__
- `TreeLite <https://treelite.readthedocs.io>`__

Models from other libraries are supported indirectly by using TreeLite as an intermediate
format - for example, objects from |sklearn| such as :obj:`sklearn.ensemble.HistGradientBoostingClassifier`
can be converted to ``daal4py`` by first converting them to TreeLite and then converting
the resulting object to ``daal4py``.

Acceleration is achieved by a smart arrangement of tree representations in memory which
is optimized for the way in which modern CPUs interact with RAM, along with leveraging of
instruction set extensions for Intel hardware; which can result in very significant speed
ups of model predictions (inference) without any loss of numerical precision compared to
the original libraries that produced the models.

In addition to regular predictions, ``daal4py`` can also accelerate SHAP computations,
for both feature contributions and feature interactions.

Example
-------

Example converting an XGBoost model:

.. code-block:: python

    import numpy as np
    import xgboost as xgb
    import daal4py
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=100, n_features=10, random_state=123)
    dm = xgb.DMatrix(X, y)
    xgb_model = xgb.train(
        params={"objective": "reg:squarederror"},
        dtrain=dm,
        num_boost_round=10
    )

    d4p_model = daal4py.mb.convert_model(xgb_model)

    np.testing.assert_allclose(
        xgb_model.predict(dm),
        d4p_model.predict(X),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        xgb_model.predict(dm, pred_contribs=True),
        d4p_model.predict(X, pred_contribs=True),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        xgb_model.predict(dm, pred_interactions=True),
        d4p_model.predict(X, pred_interactions=True),
        rtol=1e-6,
    )

Example converting a |sklearn| model using TreeLite as intermediate format:

.. code-block:: python

    import numpy as np
    import treelite
    import daal4py
    from sklearn.datasets import make_regression
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y = make_regression(n_samples=100, n_features=10, random_state=123)
    skl_model = HistGradientBoostingRegressor(max_iter=5).fit(X, y)
    tl_model = treelite.sklearn.import_model(skl_model)

    d4p_model = daal4py.mb.convert_model(tl_model)

    np.testing.assert_allclose(
        d4p_model.predict(X),
        treelite.gtil.predict(tl_model, X).reshape(-1),
        rtol=1e-6,
    )


More examples:

- `XGBoost model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_xgboost.py>`__
- `SHAP value prediction from an XGBoost model <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_xgboost_shap.py>`__
- `LightGBM model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_lightgbm.py>`__
- `CatBoost model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_catboost.py>`__

Blog posts:

- `Faster XGBoost, LightGBM, and CatBoost Inference on the CPU <https://www.intel.com/content/www/us/en/developer/articles/technical/faster-xgboost-light-gbm-catboost-inference-on-cpu.html>`__
- `Improving the Performance of XGBoost and LightGBM Inference <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>`__

Notes about computed results
----------------------------

- The shapes of SHAP contributions and interactions are consistent with the XGBoost results.
  In contrast, the `SHAP Python package <https://shap.readthedocs.io/en/latest/>`_ drops bias terms, resulting
  in SHAP contributions (SHAP interactions) with one fewer column (one fewer column and row) per observation.
- Predictions for regression objectives are computed in the link scale only (what XGBoost calls "margin" and
  LightGBM calls "raw").

Limitations
-----------

- Models with categorical features are not supported.
- Multi-class classification is only supported when the logic corresponds to multinomial logistic loss
  instead of one-vs-rest.
- Multioutput models are not supported.
- SHAP values cannot be calculated for multi-class classification models, nor for CatBoost regression models
  from loss functions that involve link functions (e.g. can be calculated for 'RMSE', but not for 'Poisson').
- Objectives that are not for regression nor classification (e.g. ranking) are not supported.
- Random forests converted to TreeLite can be supported when they are for regression or binary classification,
  but not when they are for multi-class classification. In the case of binary classification, random forests
  are converted as regression models since they do not apply a link function to predictions the same way
  gradient boosting models do.

Documentation
-------------

See the section about :ref:`model builders <model_builders_docs>` in the ``daal4py`` API reference
for full documentation.
