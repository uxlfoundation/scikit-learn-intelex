.. Copyright 2023 Intel Corporation
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

.. _model-builders:

###################################################
Model Builders for the Gradient Boosting Frameworks
###################################################

.. include:: note.rst

Introduction
------------------
Gradient boosting on decision trees is one of the most accurate and efficient
machine learning algorithms for classification and regression.
The most popular implementations of it are:

* XGBoost*
* LightGBM*
* CatBoost*

daal4py Model Builders deliver the accelerated
models inference of those frameworks. The inference is performed by the oneDAL GBT implementation.

.. note::

   Currently, experimental support for XGBoost* and LightGBM* categorical data is not supported.
   For the model conversion to work with daal4py, convert non-numeric data to numeric data
   before training and converting the model.

Conversion
----------
The first step is to convert already trained model. The
API usage for different frameworks is the same:

XGBoost::

  import daal4py as d4p
  d4p_model = d4p.mb.convert_model(xgb_model)

LightGBM::

  import daal4py as d4p
  d4p_model = d4p.mb.convert_model(lgb_model)

CatBoost::

  import daal4py as d4p
  d4p_model = d4p.mb.convert_model(cb_model)

.. note:: Convert model only once and then use it for the inference.

Classification and Regression Inference
----------------------------------------

The API is the same for classification and regression inference.
Based on the original model passed to the ``convert_model()``, ``d4p_prediction`` is either the classification or regression output.

    ::

      d4p_prediction = d4p_model.predict(test_data)

Here, the ``predict()`` method of ``d4p_model`` is being used to make predictions on the ``test_data`` dataset.
The ``d4p_prediction`` variable stores the predictions made by the ``predict()`` method.

SHAP Value Calculation for Regression Models
------------------------------------------------------------

SHAP contribution and interaction value calculation are natively supported by models created with daal4py Model Builders.
For these models, the ``predict()`` method takes additional keyword arguments:

    ::

      d4p_model.predict(test_data, pred_contribs=True)      # for SHAP contributions
      d4p_model.predict(test_data, pred_interactions=True)  # for SHAP interactions

The returned prediction has the shape:

   * ``(n_rows, n_features + 1)``  for SHAP contributions
   * ``(n_rows, n_features + 1, n_features + 1)`` for SHAP interactions

Here, ``n_rows`` is the number of rows (i.e., observations) in
``test_data``, and ``n_features`` is the number of features in the dataset.

The prediction result for SHAP contributions includes a feature attribution value for each feature and a bias term for each observation.

The prediction result for SHAP interactions comprises ``(n_features + 1) x (n_features + 1)`` values for all possible
feature combinations, along with their corresponding bias terms.

.. note:: The shapes of SHAP contributions and interactions are consistent with the XGBoost results.
  In contrast, the `SHAP Python package <https://shap.readthedocs.io/en/latest/>`_ drops bias terms, resulting
  in SHAP contributions (SHAP interactions) with one fewer column (one fewer column and row) per observation.

Scikit-learn-style Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use the scikit-learn-style classes ``GBTDAALClassifier`` and ``GBTDAALRegressor`` to convert and infer your models. For example:

::

  from daal4py.sklearn.ensemble import GBTDAALRegressor
  reg = xgb.XGBRegressor()
  reg.fit(X, y)
  d4p_predt = GBTDAALRegressor.convert_model(reg).predict(X)


Limitations
------------------
Model Builders support only base inference with prediction and probabilities prediction. The functionality is to be extended.
Therefore, there are the following limitations:
- The categorical features are not supported for conversion and prediction.
- The multioutput models are not supported for conversion and prediction.
- SHAP values can be calculated for regression models only.


Examples
---------------------------------
Model Builders models conversion

- `XGBoost model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/daal4py/model_builders_xgboost.py>`_
- `SHAP value prediction from an XGBoost model <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/daal4py/model_builders_xgboost_shap.py>`_
- `LightGBM model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/daal4py/model_builders_lightgbm.py>`_
- `CatBoost model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/daal4py/model_builders_catboost.py>`_

Articles and Blog Posts
---------------------------------

-  `Improving the Performance of XGBoost and LightGBM Inference <https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e>`_

