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

##########
Examples
##########

.. include:: note.rst

Below are examples on how to utilize daal4py for various usage styles.

General usage
-------------

Building models from Gradient Boosting frameworks

- `XGBoost* model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_xgboost.py>`_
- `LightGBM* model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_lightgbm.py>`_
- `CatBoost* model conversion <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/model_builders_catboost.py>`_


Principal Component Analysis (PCA) Transform

- `Single-Process PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca.py>`_
- `Multi-Process  PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_spmd.py>`_

Singular Value Decomposition (SVD)

- `Single-Process PCA Transform <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_transform.py>`_

- `Single-Process SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd.py>`_
- `Streaming SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_streaming.py>`_
- `Multi-Process SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_spmd.py>`_

Moments of Low Order

- `Single-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_dense.py>`_
- `Streaming Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_dense_streaming.py>`_
- `Multi-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_spmd.py>`_

Correlation and Variance-Covariance Matrices

- `Single-Process Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance.py>`_
- `Streaming Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_streaming.py>`_
- `Multi-Process Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_spmd.py>`_

Decision Forest Classification

- `Single-Process Decision Forest Classification Default Dense method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_classification_default_dense.py>`_
- `Single-Process Decision Forest Classification Histogram method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_classification_hist.py>`_

Decision Tree Classification

- `Single-Process Decision Tree Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_classification.py>`_

Gradient Boosted Classification

- `Single-Process Gradient Boosted Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_classification.py>`_

k-Nearest Neighbors (kNN)

- `Single-Process kNN
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kdtree_knn_classification.py>`_

Multinomial Naive Bayes

- `Single-Process Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes.py>`_
- `Streaming Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_streaming.py>`_
- `Multi-Process  Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_spmd.py>`_

Support Vector Machine (SVM)

- `Single-Process Binary SVM
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svm.py>`_

- `Single-Process Muticlass SVM
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svm_multiclass.py>`_

Logistic Regression

- `Single-Process Binary Class Logistic Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/log_reg_binary_dense.py>`_
- `Single-Process Logistic Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/log_reg_dense.py>`_

Decision Forest Regression

- `Single-Process Decision Forest Regression Default Dense method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_regression_default_dense.py>`_
- `Single-Process Decision Forest Regression Histogram method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_regression_hist.py>`_

- `Single-Process Decision Tree Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_regression.py>`_

Gradient Boosted Regression

- `Single-Process Boosted Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_regression.py>`_

Linear Regression

- `Single-Process Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression.py>`_
- `Streaming Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_streaming.py>`_
- `Multi-Process Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_spmd.py>`_

Ridge Regression

- `Single-Process Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression.py>`_
- `Streaming Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_streaming.py>`_
- `Multi-Process Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_spmd.py>`_

K-Means Clustering

- `Single-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans.py>`_
- `Multi-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans_spmd.py>`_

Multivariate Outlier Detection

- `Single-Process Multivariate Outlier Detection <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/multivariate_outlier.py>`_

Univariate Outlier Detection

- `Single-Process Univariate Outlier Detection <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/univariate_outlier.py>`_

Optimization Solvers-Mean Squared Error Algorithm (MSE)

- `MSE In Adagrad <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/adagrad_mse.py>`_
- `MSE In LBFGS <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_mse.py>`_
- `MSE In SGD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_mse.py>`_

Logistic Loss

- `Logistic Loss SGD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_logistic_loss.py>`_

Stochastic Gradient Descent Algorithm

- `Stochastic Gradient Descent Algorithm Using Logistic Loss <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_logistic_loss.py>`_
- `Stochastic Gradient Descent Algorithm Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_mse.py>`_

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm

- `Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm - Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_mse.py>`_

Adaptive Subgradient Method

- `Adaptive Subgradient Method Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/adagrad_mse.py>`_

Cosine Distance Matrix

- `Single-Process Cosine Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/cosine_distance.py>`_

Correlation Distance Matrix

- `Single-Process Correlation Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/correlation_distance.py>`_

Trees

- `Decision Forest Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_regression_traverse.py>`_
- `Decision Forest Classification <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_classification_traverse.py>`_
- `Decision Tree Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_regression_traverse.py>`_
- `Decision Tree Classification <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_classification_traverse.py>`_
- `Gradient Boosted Trees Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_regression_traverse.py>`_
- `Gradient Boosted Trees Classification <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_classification_traverse.py>`_
