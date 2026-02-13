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
.. _daal4py_ref:

daal4py API Reference
=====================

This is the full documentation page for ``daal4py`` functions and classes. Note that for the most
part, these are simple wrappers over equivalent functions and methods from the |onedal|. See also
the :external+onedal:doc:`documentation of DAAL interfaces <daal-interfaces>` for more details.

See :ref:`about_daal4py` for an example of how to use ``daal4py`` algorithms.

Thread control
--------------

Documentation for functions that control the global thread settings in ``daal4py``:

.. autofunction:: daal4py.daalinit
.. autofunction:: daal4py.num_threads
.. autofunction:: daal4py.enable_thread_pinning

MPI helpers
-----------

Documentation for helper functions that can be used in distributed mode, particularly when using MPI without |mpi4py|.
See :ref:`distributed_daal4py` for examples.

.. autofunction:: daal4py.daalfini
.. autofunction:: daal4py.num_procs
.. autofunction:: daal4py.my_procid


.. _model_builders_docs:

Model builders (GBT and LogReg serving)
---------------------------------------

Documentation for model builders, which allow computing fast predictions from GBT
(gradient-boosted decision tree) models produced by other libraries. See article
:ref:`model_builders` for examples.

.. autofunction:: daal4py.mb.convert_model

.. autoclass:: daal4py.mb.GBTDAALModel
  :members:
  :exclude-members: is_classifier_, is_regressor_

.. autoclass:: daal4py.mb.LogisticDAALModel
  :members:

Classification
--------------

.. note::
    All classification algorithms produce a result object of the same class, containing
    predicted probabilities, logarithm of the predicted probabilities, and most probable
    class.

Results class
*************
.. autoclass:: daal4py.classifier_prediction_result
   :members:


Decision Forest Classification
******************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/decision_forest/decision-forest-classification`.

.. rubric:: Examples:

- `Single-Process Decision Forest Classification Default Dense method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_classification_default_dense.py>`__
- `Single-Process Decision Forest Classification Histogram method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_classification_hist.py>`__

.. autoclass:: daal4py.decision_forest_classification_training
   :members: compute
.. autoclass:: daal4py.decision_forest_classification_training_result
   :members:
.. autoclass:: daal4py.decision_forest_classification_prediction
   :members: compute
.. autoclass:: daal4py.decision_forest_classification_model
   :members:

Gradient Boosted Classification
*******************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-classification`.

.. rubric:: Examples:

- `Single-Process Gradient Boosted Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_classification.py>`__

.. autoclass:: daal4py.gbt_classification_training
   :members: compute
.. autoclass:: daal4py.gbt_classification_training_result
   :members:
.. autoclass:: daal4py.gbt_classification_prediction
   :members: compute
.. autoclass:: daal4py.gbt_classification_model
   :members:

k-Nearest Neighbors (kNN)
*************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier`.

.. rubric:: Examples:

- `Single-Process kNN
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kdtree_knn_classification.py>`__

.. autoclass:: daal4py.kdtree_knn_classification_training
   :members: compute
.. autoclass:: daal4py.kdtree_knn_classification_training_result
   :members:
.. autoclass:: daal4py.kdtree_knn_classification_prediction
   :members: compute
.. autoclass:: daal4py.kdtree_knn_classification_model
   :members:

Brute-force k-Nearest Neighbors (kNN)
*************************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier`.

.. autoclass:: daal4py.bf_knn_classification_training
   :members: compute
.. autoclass:: daal4py.bf_knn_classification_training_result
   :members:
.. autoclass:: daal4py.bf_knn_classification_prediction
   :members: compute
.. autoclass:: daal4py.bf_knn_classification_model
   :members:

Support Vector Machine (SVM)
****************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/svm/support-vector-machine-classifier`.

Note: For the labels parameter, data is formatted as -1s and 1s

.. rubric:: Examples:

- `Single-Process SVM
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svm.py>`__

.. autoclass:: daal4py.svm_training
   :members: compute
.. autoclass:: daal4py.svm_training_result
   :members:
.. autoclass:: daal4py.svm_prediction
   :members: compute
.. autoclass:: daal4py.svm_model
   :members:

Logistic Regression
*******************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/logistic_regression/logistic-regression`.

.. rubric:: Examples:

- `Single-Process Binary Class Logistic Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/log_reg_binary_dense.py>`__
- `Single-Process Logistic Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/log_reg_dense.py>`__

.. autoclass:: daal4py.logistic_regression_training
   :members: compute
.. autoclass:: daal4py.logistic_regression_training_result
   :members:
.. autoclass:: daal4py.logistic_regression_prediction
   :members: compute
.. autoclass:: daal4py.logistic_regression_model
   :members:

Regression
----------

Decision Forest Regression
**************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/decision_forest/decision-forest-regression`.

.. rubric:: Examples:

- `Single-Process Decision Forest Regression Default Dense method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_regression_default_dense.py>`__
- `Single-Process Decision Forest Regression Histogram method
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_forest_regression_hist.py>`__

.. autoclass:: daal4py.decision_forest_regression_training
   :members: compute
.. autoclass:: daal4py.decision_forest_regression_training_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_prediction
   :members: compute
.. autoclass:: daal4py.decision_forest_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_forest_regression_model
   :members:

Gradient Boosted Regression
***************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-regression`.

.. rubric:: Examples:

- `Single-Process Boosted Regression Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/gradient_boosted_regression.py>`__

.. autoclass:: daal4py.gbt_regression_training
   :members: compute
.. autoclass:: daal4py.gbt_regression_training_result
   :members:
.. autoclass:: daal4py.gbt_regression_prediction
   :members: compute
.. autoclass:: daal4py.gbt_regression_prediction_result
   :members:
.. autoclass:: daal4py.gbt_regression_model
   :members:

Linear Regression
*****************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/linear_ridge_regression/linear-regression`.

.. rubric:: Examples:

- `Single-Process Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression.py>`__
- `Streaming Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_streaming.py>`__
- `Multi-Process Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_spmd.py>`__

.. autoclass:: daal4py.linear_regression_training
   :members: compute
.. autoclass:: daal4py.linear_regression_training_result
   :members:
.. autoclass:: daal4py.linear_regression_prediction
   :members: compute
.. autoclass:: daal4py.linear_regression_prediction_result
   :members:
.. autoclass:: daal4py.linear_regression_model
   :members:

LASSO Regression
****************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/lasso_elastic_net/lasso`.

.. rubric:: Examples:

- `Single-Process LASSO Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lasso_regression.py>`__

.. autoclass:: daal4py.lasso_regression_training
   :members: compute
.. autoclass:: daal4py.lasso_regression_training_result
   :members:
.. autoclass:: daal4py.lasso_regression_prediction
   :members: compute
.. autoclass:: daal4py.lasso_regression_prediction_result
   :members:
.. autoclass:: daal4py.lasso_regression_model
   :members:

ElasticNet Regression
*********************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/lasso_elastic_net/elastic-net`.

.. rubric:: Examples:

- `Single-Process ElasticNet Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/elasticnet.py>`__

.. autoclass:: daal4py.elastic_net_training
   :members: compute
.. autoclass:: daal4py.elastic_net_training_result
   :members:
.. autoclass:: daal4py.elastic_net_prediction
   :members: compute
.. autoclass:: daal4py.elastic_net_prediction_result
   :members:
.. autoclass:: daal4py.elastic_net_model
   :members:

Ridge Regression
****************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/linear_ridge_regression/ridge-regression`.

.. rubric:: Examples:

- `Single-Process Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression.py>`__
- `Streaming Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_streaming.py>`__
- `Multi-Process Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_spmd.py>`__

.. autoclass:: daal4py.ridge_regression_training
   :members: compute
.. autoclass:: daal4py.ridge_regression_training_result
   :members:
.. autoclass:: daal4py.ridge_regression_prediction
   :members: compute
.. autoclass:: daal4py.ridge_regression_prediction_result
   :members:
.. autoclass:: daal4py.ridge_regression_model
   :members:

Clustering
----------

K-Means Clustering
******************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/kmeans/k-means-clustering`.

.. rubric:: Examples:

- `Single-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans.py>`__
- `Multi-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans_spmd.py>`__

K-Means Initialization
**********************
Parameters and semantics are described in :ref:`oneAPI Data Analytics Library K-Means Initialization <onedal:kmeans_init>`

.. autoclass:: daal4py.kmeans_init
   :members: compute
.. autoclass:: daal4py.kmeans_init_result
   :members:

K-Means
*******
Parameters and semantics are described in |onedal-dg-k-means-computation|_.

.. autoclass:: daal4py.kmeans
   :members: compute
.. autoclass:: daal4py.kmeans_result
   :members:

DBSCAN
******
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/dbscan/index`.

.. rubric:: Examples:

- `Single-Process DBSCAN <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/dbscan.py>`__

.. autoclass:: daal4py.dbscan
   :members: compute
.. autoclass:: daal4py.dbscan_result
   :members:

Gaussian Mixtures
*****************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/em/expectation-maximization`.

Initialization for the Gaussian Mixture Model
"""""""""""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-expectation-maximization-initialization|_.

.. rubric:: Examples:

- `Single-Process Expectation-Maximization <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/em_gmm.py>`__

.. autoclass:: daal4py.em_gmm_init
   :members: compute
.. autoclass:: daal4py.em_gmm_init_result
   :members:

EM algorithm for the Gaussian Mixture Model
"""""""""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-expectation-maximization-for-the-gaussian-mixture-model|_.

.. rubric:: Examples:

- `Single-Process Expectation-Maximization <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/em_gmm.py>`__

.. autoclass:: daal4py.em_gmm
   :members: compute
.. autoclass:: daal4py.em_gmm_result
   :members:

Dimensionality reduction
------------------------

Principal Component Analysis (PCA)
**********************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/pca/principal-component-analysis`.

.. rubric:: Examples:

- `Single-Process PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca.py>`__
- `Multi-Process PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_spmd.py>`__

.. autoclass:: daal4py.pca
   :members: compute
.. autoclass:: daal4py.pca_result
   :members:

Principal Component Analysis (PCA) Transform
********************************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/pca/transform`.

.. rubric:: Examples:

- `Single-Process PCA Transform <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_transform.py>`__

.. autoclass:: daal4py.pca_transform
   :members: compute
.. autoclass:: daal4py.pca_transform_result
   :members:

Optimization Solvers
--------------------

Objective Functions
*******************

Mean Squared Error Algorithm (MSE)
""""""""""""""""""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/objective-functions/mse`.

.. rubric:: Examples:

- `In Adagrad <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/adagrad_mse.py>`__
- `In LBFGS <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_mse.py>`__
- `In SGD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_mse.py>`__

.. autoclass:: daal4py.optimization_solver_mse
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_mse_result
   :members:

Logistic Loss
"""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/objective-functions/logistic-loss`.

.. rubric:: Examples:

- `In SGD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_logistic_loss.py>`__

.. autoclass:: daal4py.optimization_solver_logistic_loss
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_logistic_loss_result
   :members:

Cross-entropy Loss
""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/objective-functions/cross-entropy`.

.. rubric:: Examples:

- `In LBFGS <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_cr_entr_loss.py>`__

.. autoclass:: daal4py.optimization_solver_cross_entropy_loss
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_cross_entropy_loss_result
   :members:

Sum of Functions
""""""""""""""""

.. autoclass:: daal4py.optimization_solver_sum_of_functions_result
   :members:

Iterative Solvers
*****************

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/solvers/lbfgs`.

.. rubric:: Examples:

- `Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_mse.py>`__

.. autoclass:: daal4py.optimization_solver_lbfgs
   :members: compute
.. autoclass:: daal4py.optimization_solver_lbfgs_result
   :members:

Coordinate Descent
""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/solvers/coordinate-descent`.

.. autoclass:: daal4py.optimization_solver_coordinate_descent
   :members: compute
.. autoclass:: daal4py.optimization_solver_coordinate_descent_result
   :members:

Precomputed Function
""""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/optimization-solvers/objective-functions/with-precomputed-characteristics`.

.. autoclass:: daal4py.optimization_solver_precomputed
   :members: compute
.. autoclass:: daal4py.optimization_solver_precomputed_result
   :members:

Implicit Alternating Least Squares (implicit ALS)
*************************************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/implicit_als/implicit-alternating-least-squares`.

.. rubric:: Examples:

- `Single-Process implicit ALS <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/implicit_als.py>`__

.. autoclass:: daal4py.implicit_als_training
   :members: compute
.. autoclass:: daal4py.implicit_als_training_result
   :members:
.. autoclass:: daal4py.implicit_als_model
   :members:
.. autoclass:: daal4py.implicit_als_prediction_ratings
   :members: compute
.. autoclass:: daal4py.implicit_als_prediction_ratings_result
   :members:

Covariance, correlation, and distances
--------------------------------------

Cosine Distance Matrix
**********************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distance/cosine`.

.. rubric:: Examples:

- `Single-Process Cosine Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/cosine_distance.py>`__

.. autoclass:: daal4py.cosine_distance
   :members: compute
.. autoclass:: daal4py.cosine_distance_result
   :members:

Correlation Distance Matrix
***************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distance/correlation`.

.. rubric:: Examples:

- `Single-Process Correlation Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/correlation_distance.py>`__

.. autoclass:: daal4py.correlation_distance
   :members: compute
.. autoclass:: daal4py.correlation_distance_result
   :members:

Correlation and Variance-Covariance Matrices
********************************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/covariance/correlation-and-variance-covariance-matrices`.

.. rubric:: Examples:

- `Single-Process Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance.py>`__
- `Streaming Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_streaming.py>`__
- `Multi-Process Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_spmd.py>`__

.. autoclass:: daal4py.covariance
   :members: compute
.. autoclass:: daal4py.covariance_result
   :members:

Data pre-processing
-------------------

Normalization
*************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/normalization/index`.

Z-Score
"""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/normalization/z-score`.

.. rubric:: Examples:

- `Single-Process Z-Score Normalization <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/normalization_zscore.py>`__

.. autoclass:: daal4py.normalization_zscore
   :members: compute
.. autoclass:: daal4py.normalization_zscore_result
   :members:

Min-Max
"""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/normalization/min-max`.

.. rubric:: Examples:

- `Single-Process Min-Max Normalization <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/normalization_minmax.py>`__

.. autoclass:: daal4py.normalization_minmax
   :members: compute
.. autoclass:: daal4py.normalization_minmax_result
   :members:

Statistics
----------

Moments of Low Order
********************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/moments/moments-of-low-order`.

.. rubric:: Examples:

- `Single-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_dense.py>`__
- `Streaming Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_streaming.py>`__
- `Multi-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_spmd.py>`__

.. autoclass:: daal4py.low_order_moments
   :members: compute
.. autoclass:: daal4py.low_order_moments_result
   :members:

Linear algebra
--------------

Cholesky Decomposition
**********************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/cholesky/cholesky`.

.. rubric:: Examples:

- `Single-Process Cholesky <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/cholesky.py>`__

.. autoclass:: daal4py.cholesky
   :members: compute
.. autoclass:: daal4py.cholesky_result
   :members:

QR Decomposition
****************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/qr/qr-decomposition`.

QR Decomposition (without pivoting)
"""""""""""""""""""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/qr/qr-without-pivoting`.

.. rubric:: Examples:

- `Single-Process QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr.py>`__
- `Streaming QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr_streaming.py>`__

.. autoclass:: daal4py.qr
   :members: compute
.. autoclass:: daal4py.qr_result
   :members:

Pivoted QR Decomposition
""""""""""""""""""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/qr/qr-pivoted`.

.. rubric:: Examples:

- `Single-Process Pivoted QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pivoted_qr.py>`__

.. autoclass:: daal4py.pivoted_qr
   :members: compute
.. autoclass:: daal4py.pivoted_qr_result
   :members:

Singular Value Decomposition (SVD)
**********************************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/svd/singular-value-decomposition`.

.. rubric:: Examples:

- `Single-Process SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd.py>`__
- `Streaming SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_streaming.py>`__
- `Multi-Process SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_spmd.py>`__

.. autoclass:: daal4py.svd
   :members: compute
.. autoclass:: daal4py.svd_result
   :members:

Random number generation
------------------------

Random Number Engines
*********************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/engines/index`.

.. autoclass:: daal4py.engines_result
   :members:

mt19937
"""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/engines/mt19937`.

.. autoclass:: daal4py.engines_mt19937
   :members: compute
.. autoclass:: daal4py.engines_mt19937_result
   :members:

mt2203
""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/engines/mt2203`.

.. autoclass:: daal4py.engines_mt2203
   :members: compute
.. autoclass:: daal4py.engines_mt2203_result
   :members:

mcg59
"""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/engines/mcg59`.

.. autoclass:: daal4py.engines_mcg59
   :members: compute
.. autoclass:: daal4py.engines_mcg59_result
   :members:

mrg32k3a
""""""""

.. autoclass:: daal4py.engines_mrg32k3a
   :members: compute
.. autoclass:: daal4py.engines_mrg32k3a_result
   :members:

philox4x32x10
"""""""""""""

.. autoclass:: daal4py.engines_philox4x32x10
   :members: compute
.. autoclass:: daal4py.engines_philox4x32x10_result
   :members:

Distributions
*************
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distributions/index`.

Bernoulli
"""""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distributions/bernoulli`.

.. rubric:: Examples:

- `Single-Process Bernoulli Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_bernoulli.py>`__

.. autoclass:: daal4py.distributions_bernoulli
   :members: compute
.. autoclass:: daal4py.distributions_bernoulli_result
   :members:

Normal
""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distributions/normal`.

.. rubric:: Examples:

- `Single-Process Normal Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_normal.py>`__

.. autoclass:: daal4py.distributions_normal
   :members: compute
.. autoclass:: daal4py.distributions_normal_result
   :members:

Uniform
"""""""
Parameters and semantics are described in :external+onedal:doc:`daal/algorithms/distributions/uniform`.

.. rubric:: Examples:

- `Single-Process Uniform Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_uniform.py>`__

.. autoclass:: daal4py.distributions_uniform
   :members: compute
.. autoclass:: daal4py.distributions_uniform_result
   :members:

.. Note: oneDAL's doc do not have any named object referencing these specific sections,
.. hence the need for this workaround.
.. |onedal-dg-k-means-computation| replace:: |onedal| K-Means Computation
.. _onedal-dg-k-means-computation: https://uxlfoundation.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html#computation

.. |onedal-dg-expectation-maximization-initialization| replace:: |onedal| Expectation-Maximization Initialization
.. _onedal-dg-expectation-maximization-initialization: https://uxlfoundation.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#initialization

.. |onedal-dg-expectation-maximization-for-the-gaussian-mixture-model| replace:: |onedal| Expectation-Maximization for the Gaussian Mixture Model
.. _onedal-dg-expectation-maximization-for-the-gaussian-mixture-model: https://uxlfoundation.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#em-algorithm-for-the-gaussian-mixture-model
