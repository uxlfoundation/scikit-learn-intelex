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
the `documentation of DAAL interfaces <https://uxlfoundation.github.io/oneDAL/daal-interfaces.html>`__ for more details.

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
Parameters and semantics are described in |onedal-dg-classification-decision-forest|_.

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

Decision Tree Classification
****************************
Parameters and semantics are described in |onedal-dg-classification-decision-tree|_.

.. rubric:: Examples:

- `Single-Process Decision Tree Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_classification.py>`__

.. autoclass:: daal4py.decision_tree_classification_training
   :members: compute
.. autoclass:: daal4py.decision_tree_classification_training_result
   :members:
.. autoclass:: daal4py.decision_tree_classification_prediction
   :members: compute
.. autoclass:: daal4py.decision_tree_classification_model
   :members:

Gradient Boosted Classification
*******************************
Parameters and semantics are described in |onedal-dg-classification-gradient-boosted-tree|_.

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
Parameters and semantics are described in |onedal-dg-k-nearest-neighbors-knn|_.

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
Parameters and semantics are described in |onedal-dg-k-nearest-neighbors-knn|_.

.. autoclass:: daal4py.bf_knn_classification_training
   :members: compute
.. autoclass:: daal4py.bf_knn_classification_training_result
   :members:
.. autoclass:: daal4py.bf_knn_classification_prediction
   :members: compute
.. autoclass:: daal4py.bf_knn_classification_model
   :members:

AdaBoost Classification
***********************
Parameters and semantics are described in |onedal-dg-classification-adaboost|_.

.. rubric:: Examples:

- `Single-Process AdaBoost Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/adaboost.py>`__

.. autoclass:: daal4py.adaboost_training
   :members: compute
.. autoclass:: daal4py.adaboost_training_result
   :members:
.. autoclass:: daal4py.adaboost_prediction
   :members: compute
.. autoclass:: daal4py.adaboost_model
   :members:

BrownBoost Classification
*************************
Parameters and semantics are described in |onedal-dg-classification-brownboost|_.

.. rubric:: Examples:

- `Single-Process BrownBoost Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/brownboost.py>`__

.. autoclass:: daal4py.brownboost_training
   :members: compute
.. autoclass:: daal4py.brownboost_training_result
   :members:
.. autoclass:: daal4py.brownboost_prediction
   :members: compute
.. autoclass:: daal4py.brownboost_model
   :members:

LogitBoost Classification
*************************
Parameters and semantics are described in |onedal-dg-classification-logitboost|_.

.. rubric:: Examples:

- `Single-Process LogitBoost Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/logitboost.py>`__

.. autoclass:: daal4py.logitboost_training
   :members: compute
.. autoclass:: daal4py.logitboost_training_result
   :members:
.. autoclass:: daal4py.logitboost_prediction
   :members: compute
.. autoclass:: daal4py.logitboost_model
   :members:

Stump Weak Learner Classification
*********************************
Parameters and semantics are described in |onedal-dg-classification-weak-learner-stump|_.

.. rubric:: Examples:

- `Single-Process Stump Weak Learner Classification
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/stump_classification.py>`__

.. autoclass:: daal4py.stump_classification_training
   :members: compute
.. autoclass:: daal4py.stump_classification_training_result
   :members:
.. autoclass:: daal4py.stump_classification_prediction
   :members: compute
.. autoclass:: daal4py.stump_classification_model
   :members:

Multinomial Naive Bayes
***********************
Parameters and semantics are described in |onedal-dg-naive-bayes|_.

.. rubric:: Examples:

- `Single-Process Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes.py>`__
- `Streaming Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_streaming.py>`__
- `Multi-Process Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_spmd.py>`__

.. autoclass:: daal4py.multinomial_naive_bayes_training
   :members: compute
.. autoclass:: daal4py.multinomial_naive_bayes_training_result
   :members:
.. autoclass:: daal4py.multinomial_naive_bayes_prediction
   :members: compute
.. autoclass:: daal4py.multinomial_naive_bayes_model
   :members:

Support Vector Machine (SVM)
****************************
Parameters and semantics are described in |onedal-dg-svm|_.

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
Parameters and semantics are described in |onedal-dg-logistic-regression|_.

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
Parameters and semantics are described in |onedal-dg-regression-decision-forest|_.

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

Decision Tree Regression
************************
Parameters and semantics are described in |onedal-dg-regression-decision-tree|_.

.. rubric:: Examples:

- `Single-Process Decision Tree Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/decision_tree_regression.py>`__

.. autoclass:: daal4py.decision_tree_regression_training
   :members: compute
.. autoclass:: daal4py.decision_tree_regression_training_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_prediction
   :members: compute
.. autoclass:: daal4py.decision_tree_regression_prediction_result
   :members:
.. autoclass:: daal4py.decision_tree_regression_model
   :members:

Gradient Boosted Regression
***************************
Parameters and semantics are described in |onedal-dg-regression-gradient-boosted-tree|_.

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
Parameters and semantics are described in |onedal-dg-linear-regression|_.

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
Parameters and semantics are described in |onedal-dg-least-absolute-shrinkage-and-selection-operator|_.

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

Ridge Regression
****************
Parameters and semantics are described in |onedal-dg-ridge-regression|_.

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

Stump Regression
****************
Parameters and semantics are described in |onedal-dg-regression-stump|_.

.. rubric:: Examples:

- `Single-Process Stump Regression
  <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/stump_regression.py>`__

.. autoclass:: daal4py.stump_regression_training
   :members: compute
.. autoclass:: daal4py.stump_regression_training_result
   :members:
.. autoclass:: daal4py.stump_regression_prediction
   :members: compute
.. autoclass:: daal4py.stump_regression_prediction_result
   :members:
.. autoclass:: daal4py.stump_regression_model
   :members:

Clustering
----------

K-Means Clustering
******************
Parameters and semantics are described in |onedal-dg-k-means-clustering|_.

.. rubric:: Examples:

- `Single-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans.py>`__
- `Multi-Process K-Means <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/kmeans_spmd.py>`__

K-Means Initialization
**********************
Parameters and semantics are described in |onedal-dg-k-means-initialization|_.

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
Parameters and semantics are described in |onedal-dg-density-based-spatial-clustering-of-applications-with-noise|_.

.. rubric:: Examples:

- `Single-Process DBSCAN <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/dbscan.py>`__

.. autoclass:: daal4py.dbscan
   :members: compute
.. autoclass:: daal4py.dbscan_result
   :members:

Gaussian Mixtures
*****************
Parameters and semantics are described in |onedal-dg-expectation-maximization|_.

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
Parameters and semantics are described in |onedal-dg-pca|_.

.. rubric:: Examples:

- `Single-Process PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca.py>`__
- `Multi-Process PCA <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_spmd.py>`__

.. autoclass:: daal4py.pca
   :members: compute
.. autoclass:: daal4py.pca_result
   :members:

Principal Component Analysis (PCA) Transform
********************************************
Parameters and semantics are described in |onedal-dg-pca-transform|_.

.. rubric:: Examples:

- `Single-Process PCA Transform <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pca_transform.py>`__

.. autoclass:: daal4py.pca_transform
   :members: compute
.. autoclass:: daal4py.pca_transform_result
   :members:

Outlier detection
-----------------

Multivariate Outlier Detection
******************************
Parameters and semantics are described in |onedal-dg-multivariate-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Multivariate Outlier Detection <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/multivariate_outlier.py>`__

.. autoclass:: daal4py.multivariate_outlier_detection
   :members: compute
.. autoclass:: daal4py.multivariate_outlier_detection_result
   :members:

Univariate Outlier Detection
****************************
Parameters and semantics are described in |onedal-dg-univariate-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Univariate Outlier Detection <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/univariate_outlier.py>`__

.. autoclass:: daal4py.univariate_outlier_detection
   :members: compute
.. autoclass:: daal4py.univariate_outlier_detection_result
   :members:

Multivariate Bacon Outlier Detection
************************************
Parameters and semantics are described in |onedal-dg-multivariate-bacon-outlier-detection|_.

.. rubric:: Examples:

- `Single-Process Bacon Outlier Detection <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/bacon_outlier.py>`__

.. autoclass:: daal4py.bacon_outlier_detection
   :members: compute
.. autoclass:: daal4py.bacon_outlier_detection_result
   :members:

Optimization Solvers
--------------------

Objective Functions
*******************

Mean Squared Error Algorithm (MSE)
""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-mse|_.

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
Parameters and semantics are described in |onedal-dg-logistic-loss|_.

.. rubric:: Examples:

- `In SGD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_logistic_loss.py>`__

.. autoclass:: daal4py.optimization_solver_logistic_loss
   :members: compute, setup
.. autoclass:: daal4py.optimization_solver_logistic_loss_result
   :members:

Cross-entropy Loss
""""""""""""""""""
Parameters and semantics are described in |onedal-dg-cross-entropy-loss|_.

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

Stochastic Gradient Descent Algorithm
"""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-sgd|_.

.. rubric:: Examples:

- `Using Logistic Loss <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_logistic_loss.py>`__
- `Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sgd_mse.py>`__

.. autoclass:: daal4py.optimization_solver_sgd
   :members: compute
.. autoclass:: daal4py.optimization_solver_sgd_result
   :members:

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-lbfgs|_.

.. rubric:: Examples:

- `Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/lbfgs_mse.py>`__

.. autoclass:: daal4py.optimization_solver_lbfgs
   :members: compute
.. autoclass:: daal4py.optimization_solver_lbfgs_result
   :members:

Adaptive Subgradient Method
"""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-adagrad|_.

.. rubric:: Examples:

- `Using MSE <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/adagrad_mse.py>`__

.. autoclass:: daal4py.optimization_solver_adagrad
   :members: compute
.. autoclass:: daal4py.optimization_solver_adagrad_result
   :members:

Stochastic Average Gradient Descent
"""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-stochastic-average-gradient-descent-saga|_.

.. rubric:: Examples:

- `Single Proces saga-logistc_loss <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/saga.py>`__

.. autoclass:: daal4py.optimization_solver_saga
   :members: compute
.. autoclass:: daal4py.optimization_solver_saga_result
   :members:

Coordinate Descent
""""""""""""""""""
Parameters and semantics are described in |onedal-dg-coordinate-descent|_.

.. autoclass:: daal4py.optimization_solver_coordinate_descent
   :members: compute
.. autoclass:: daal4py.optimization_solver_coordinate_descent_result
   :members:

Precomputed Function
""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-precomputed-objective-function|_.

.. autoclass:: daal4py.optimization_solver_precomputed
   :members: compute
.. autoclass:: daal4py.optimization_solver_precomputed_result
   :members:

Recommender systems
-------------------

Association Rules
*****************
Parameters and semantics are described in |onedal-dg-association-rules|_.

.. rubric:: Examples:

- `Single-Process Association Rules <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/association_rules.py>`__

.. autoclass:: daal4py.association_rules
   :members: compute
.. autoclass:: daal4py.association_rules_result
   :members:

Implicit Alternating Least Squares (implicit ALS)
*************************************************
Parameters and semantics are described in |onedal-dg-implicit-alternating-least-squares|_.

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
Parameters and semantics are described in |onedal-dg-cosine-distance|_.

.. rubric:: Examples:

- `Single-Process Cosine Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/cosine_distance.py>`__

.. autoclass:: daal4py.cosine_distance
   :members: compute
.. autoclass:: daal4py.cosine_distance_result
   :members:

Correlation Distance Matrix
***************************
Parameters and semantics are described in |onedal-dg-correlation-distance|_.

.. rubric:: Examples:

- `Single-Process Correlation Distance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/correlation_distance.py>`__

.. autoclass:: daal4py.correlation_distance
   :members: compute
.. autoclass:: daal4py.correlation_distance_result
   :members:

Correlation and Variance-Covariance Matrices
********************************************
Parameters and semantics are described in |onedal-dg-correlation-and-variance-covariance-matrices|_.

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
Parameters and semantics are described in |onedal-dg-normalization|_.

Z-Score
"""""""
Parameters and semantics are described in |onedal-dg-z-score|_.

.. rubric:: Examples:

- `Single-Process Z-Score Normalization <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/normalization_zscore.py>`__

.. autoclass:: daal4py.normalization_zscore
   :members: compute
.. autoclass:: daal4py.normalization_zscore_result
   :members:

Min-Max
"""""""
Parameters and semantics are described in |onedal-dg-min-max|_.

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
Parameters and semantics are described in |onedal-dg-moments-of-low-order|_.

.. rubric:: Examples:

- `Single-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_dense.py>`__
- `Streaming Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_streaming.py>`__
- `Multi-Process Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_spmd.py>`__

.. autoclass:: daal4py.low_order_moments
   :members: compute
.. autoclass:: daal4py.low_order_moments_result
   :members:

Quantiles
*********
Parameters and semantics are described in |onedal-dg-quantiles|_.

.. rubric:: Examples:

- `Single-Process Quantiles <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/quantiles.py>`__

.. autoclass:: daal4py.quantiles
   :members: compute
.. autoclass:: daal4py.quantiles_result
   :members:

Linear algebra
--------------

Cholesky Decomposition
**********************
Parameters and semantics are described in |onedal-dg-cholesky-decomposition|_.

.. rubric:: Examples:

- `Single-Process Cholesky <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/cholesky.py>`__

.. autoclass:: daal4py.cholesky
   :members: compute
.. autoclass:: daal4py.cholesky_result
   :members:

QR Decomposition
****************
Parameters and semantics are described in |onedal-dg-qr-decomposition|_.

QR Decomposition (without pivoting)
"""""""""""""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-qr-decomposition-without-pivoting|_.

.. rubric:: Examples:

- `Single-Process QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr.py>`__
- `Streaming QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr_streaming.py>`__

.. autoclass:: daal4py.qr
   :members: compute
.. autoclass:: daal4py.qr_result
   :members:

Pivoted QR Decomposition
""""""""""""""""""""""""
Parameters and semantics are described in |onedal-dg-pivoted-qr-decomposition|_.

.. rubric:: Examples:

- `Single-Process Pivoted QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/pivoted_qr.py>`__

.. autoclass:: daal4py.pivoted_qr
   :members: compute
.. autoclass:: daal4py.pivoted_qr_result
   :members:

Singular Value Decomposition (SVD)
**********************************
Parameters and semantics are described in |onedal-dg-svd|_.

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
Parameters and semantics are described in |onedal-dg-engines|_.

.. autoclass:: daal4py.engines_result
   :members:

mt19937
"""""""
Parameters and semantics are described in |onedal-dg-mt19937|_.

.. autoclass:: daal4py.engines_mt19937
   :members: compute
.. autoclass:: daal4py.engines_mt19937_result
   :members:

mt2203
""""""
Parameters and semantics are described in |onedal-dg-mt2203|_.

.. autoclass:: daal4py.engines_mt2203
   :members: compute
.. autoclass:: daal4py.engines_mt2203_result
   :members:

mcg59
"""""
Parameters and semantics are described in |onedal-dg-mcg59|_.

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
Parameters and semantics are described in |onedal-dg-distributions|_.

Bernoulli
"""""""""
Parameters and semantics are described in |onedal-dg-bernoulli-distribution|_.

.. rubric:: Examples:

- `Single-Process Bernoulli Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_bernoulli.py>`__

.. autoclass:: daal4py.distributions_bernoulli
   :members: compute
.. autoclass:: daal4py.distributions_bernoulli_result
   :members:

Normal
""""""
Parameters and semantics are described in |onedal-dg-normal-distribution|_.

.. rubric:: Examples:

- `Single-Process Normal Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_normal.py>`__

.. autoclass:: daal4py.distributions_normal
   :members: compute
.. autoclass:: daal4py.distributions_normal_result
   :members:

Uniform
"""""""
Parameters and semantics are described in |onedal-dg-uniform-distribution|_.

.. rubric:: Examples:

- `Single-Process Uniform Distribution <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/distributions_uniform.py>`__

.. autoclass:: daal4py.distributions_uniform
   :members: compute
.. autoclass:: daal4py.distributions_uniform_result
   :members:

Sorting
-------

Parameters and semantics are described in |onedal-dg-sorting|_.

.. rubric:: Examples:

- `Single-Process Sorting <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/sorting.py>`__

.. autoclass:: daal4py.sorting
   :members: compute
.. autoclass:: daal4py.sorting_result
   :members:


.. Link replacements

.. |onedal-dg-bernoulli-distribution| replace:: |onedal| Bernoulli Distribution
.. _onedal-dg-bernoulli-distribution: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distributions/bernoulli.html

.. |onedal-dg-svd| replace:: |onedal| SVD
.. _onedal-dg-svd: https://uxlfoundation.github.io/oneDAL/daal/algorithms/svd/singular-value-decomposition.html

.. |onedal-dg-regression| replace:: |onedal| Regression
.. _onedal-dg-regression: https://uxlfoundation.github.io/oneDAL/daal/usage/training-and-prediction/regression.html

.. |onedal-dg-k-means-clustering| replace:: |onedal| K-Means Clustering
.. _onedal-dg-k-means-clustering: https://uxlfoundation.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html

.. |onedal-dg-lbfgs| replace:: |onedal| LBFGS
.. _onedal-dg-lbfgs: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/limited-memory-broyden-fletcher-goldfarb-shanno-algorithm.html

.. |onedal-dg-naive-bayes| replace:: |onedal| Naive Bayes
.. _onedal-dg-naive-bayes: https://uxlfoundation.github.io/oneDAL/daal/algorithms/naive_bayes/naive-bayes-classifier.html

.. |onedal-dg-expectation-maximization| replace:: |onedal| Expectation-Maximization
.. _onedal-dg-expectation-maximization: https://uxlfoundation.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html

.. |onedal-dg-mcg59| replace:: |onedal| mcg59
.. _onedal-dg-mcg59: https://uxlfoundation.github.io/oneDAL/daal/algorithms/engines/mcg59.html

.. |onedal-dg-least-absolute-shrinkage-and-selection-operator| replace:: |onedal| Least Absolute Shrinkage and Selection Operator
.. _onedal-dg-least-absolute-shrinkage-and-selection-operator: https://uxlfoundation.github.io/oneDAL/daal/algorithms/lasso_elastic_net/lasso.html

.. |onedal-dg-sorting| replace:: |onedal| Sorting
.. _onedal-dg-sorting: https://uxlfoundation.github.io/oneDAL/daal/algorithms/sorting/index.html

.. |onedal-dg-expectation-maximization-for-the-gaussian-mixture-model| replace:: |onedal| Expectation-Maximization for the Gaussian Mixture Model
.. _onedal-dg-expectation-maximization-for-the-gaussian-mixture-model: https://uxlfoundation.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#em-algorithm-for-the-gaussian-mixture-model

.. |onedal-dg-multivariate-outlier-detection| replace:: |onedal| Multivariate Outlier Detection
.. _onedal-dg-multivariate-outlier-detection: https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate.html

.. |onedal-dg-expectation-maximization-initialization| replace:: |onedal| Expectation-Maximization Initialization
.. _onedal-dg-expectation-maximization-initialization: https://uxlfoundation.github.io/oneDAL/daal/algorithms/em/expectation-maximization.html#initialization

.. |onedal-dg-pivoted-qr-decomposition| replace:: |onedal| Pivoted QR Decomposition
.. _onedal-dg-pivoted-qr-decomposition: https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/qr-pivoted.html

.. |onedal-dg-regression-decision-tree| replace:: |onedal| Regression Decision Tree
.. _onedal-dg-regression-decision-tree: https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-regression.html

.. |onedal-dg-k-nearest-neighbors-knn| replace:: |onedal| k-Nearest Neighbors (kNN)
.. _onedal-dg-k-nearest-neighbors-knn: https://uxlfoundation.github.io/oneDAL/daal/algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier.html

.. |onedal-dg-pca| replace:: |onedal| PCA
.. _onedal-dg-pca: https://uxlfoundation.github.io/oneDAL/daal/algorithms/pca/principal-component-analysis.html

.. |onedal-dg-sgd| replace:: |onedal| SGD
.. _onedal-dg-sgd: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-gradient-descent-algorithm.html

.. |onedal-dg-uniform-distribution| replace:: |onedal| Uniform Distribution
.. _onedal-dg-uniform-distribution: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distributions/uniform.html

.. |onedal-dg-cross-entropy-loss| replace:: |onedal| Cross Entropy Loss
.. _onedal-dg-cross-entropy-loss: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/cross-entropy.html

.. |onedal-dg-classification| replace:: |onedal| Classification
.. _onedal-dg-classification: https://uxlfoundation.github.io/oneDAL/daal/usage/training-and-prediction/classification.html

.. |onedal-dg-cosine-distance| replace:: |onedal| Cosine Distance
.. _onedal-dg-cosine-distance: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distance/cosine.html

.. |onedal-dg-regression-stump| replace:: |onedal| Regression Stump
.. _onedal-dg-regression-stump: https://uxlfoundation.github.io/oneDAL/daal/algorithms/stump/regression.html

.. |onedal-dg-multivariate-bacon-outlier-detection| replace:: |onedal| Multivariate Bacon Outlier Detection
.. _onedal-dg-multivariate-bacon-outlier-detection: https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/multivariate-bacon.html

.. |onedal-dg-logistic-regression| replace:: |onedal| Logistic Regression
.. _onedal-dg-logistic-regression: https://uxlfoundation.github.io/oneDAL/daal/algorithms/logistic_regression/logistic-regression.html

.. |onedal-dg-quantiles| replace:: |onedal| Quantiles
.. _onedal-dg-quantiles: https://uxlfoundation.github.io/oneDAL/daal/algorithms/quantiles/index.html

.. |onedal-dg-pca-transform| replace:: |onedal| PCA Transform
.. _onedal-dg-pca-transform: https://uxlfoundation.github.io/oneDAL/daal/algorithms/pca/transform.html

.. |onedal-dg-correlation-distance| replace:: |onedal| Correlation Distance
.. _onedal-dg-correlation-distance: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distance/correlation.html

.. |onedal-dg-association-rules| replace:: |onedal| Association Rules
.. _onedal-dg-association-rules: https://uxlfoundation.github.io/oneDAL/daal/algorithms/association_rules/association-rules.html

.. |onedal-dg-univariate-outlier-detection| replace:: |onedal| Univariate Outlier Detection
.. _onedal-dg-univariate-outlier-detection: https://uxlfoundation.github.io/oneDAL/daal/algorithms/outlier_detection/univariate.html

.. |onedal-dg-classification-gradient-boosted-tree| replace:: |onedal| Classification Gradient Boosted Tree
.. _onedal-dg-classification-gradient-boosted-tree: https://uxlfoundation.github.io/oneDAL/daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-classification.html

.. |onedal-dg-classification-brownboost| replace:: |onedal| Classification BrownBoost
.. _onedal-dg-classification-brownboost: https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/brownboost.html

.. |onedal-dg-regression-decision-forest| replace:: |onedal| Regression Decision Forest
.. _onedal-dg-regression-decision-forest: https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_forest/decision-forest-regression.html

.. |onedal-dg-z-score| replace:: |onedal| Z-Score
.. _onedal-dg-z-score: https://uxlfoundation.github.io/oneDAL/daal/algorithms/normalization/z-score.html

.. |onedal-dg-classification-weak-learner-stump| replace:: |onedal| Classification Weak Learner Stump
.. _onedal-dg-classification-weak-learner-stump: https://uxlfoundation.github.io/oneDAL/daal/algorithms/stump/classification.html

.. |onedal-dg-svm| replace:: |onedal| SVM
.. _onedal-dg-svm: https://uxlfoundation.github.io/oneDAL/daal/algorithms/svm/support-vector-machine-classifier.html

.. |onedal-dg-regression-gradient-boosted-tree| replace:: |onedal| Regression Gradient Boosted Tree
.. _onedal-dg-regression-gradient-boosted-tree: https://uxlfoundation.github.io/oneDAL/daal/algorithms/gradient_boosted_trees/gradient-boosted-trees-regression.html

.. |onedal-dg-logistic-loss| replace:: |onedal| Logistic Loss
.. _onedal-dg-logistic-loss: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/logistic-loss.html

.. |onedal-dg-adagrad| replace:: |onedal| AdaGrad
.. _onedal-dg-adagrad: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/adaptive-subgradient-method.html

.. |onedal-dg-qr-decomposition| replace:: |onedal| QR Decomposition
.. _onedal-dg-qr-decomposition: https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/qr-decomposition.html

.. |onedal-dg-mt19937| replace:: |onedal| mt19937
.. _onedal-dg-mt19937: https://uxlfoundation.github.io/oneDAL/daal/algorithms/engines/mt19937.html

.. |onedal-dg-implicit-alternating-least-squares| replace:: |onedal| Implicit Alternating Least Squares
.. _onedal-dg-implicit-alternating-least-squares: https://uxlfoundation.github.io/oneDAL/daal/algorithms/implicit_als/implicit-alternating-least-squares.html

.. |onedal-dg-linear-regression| replace:: |onedal| Linear Regression
.. _onedal-dg-linear-regression: https://uxlfoundation.github.io/oneDAL/daal/algorithms/linear_ridge_regression/linear-regression.html

.. |onedal-dg-classification-adaboost| replace:: |onedal| Classification AdaBoost
.. _onedal-dg-classification-adaboost: https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/adaboost.html

.. |onedal-dg-distributions| replace:: |onedal| Distributions
.. _onedal-dg-distributions: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distributions/index.html

.. |onedal-dg-correlation-and-variance-covariance-matrices| replace:: |onedal| Correlation and Variance-Covariance Matrices
.. _onedal-dg-correlation-and-variance-covariance-matrices: https://uxlfoundation.github.io/oneDAL/daal/algorithms/covariance/correlation-and-variance-covariance-matrices.html

.. |onedal-dg-classification-decision-tree| replace:: |onedal| Classification Decision Tree
.. _onedal-dg-classification-decision-tree: https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_tree/decision-tree-classification.html

.. |onedal-dg-ridge-regression| replace:: |onedal| Ridge Regression
.. _onedal-dg-ridge-regression: https://uxlfoundation.github.io/oneDAL/daal/algorithms/linear_ridge_regression/ridge-regression.html

.. |onedal-dg-classification-logitboost| replace:: |onedal| Classification LogitBoost
.. _onedal-dg-classification-logitboost: https://uxlfoundation.github.io/oneDAL/daal/algorithms/boosting/logitboost.html

.. |onedal-dg-k-means-initialization| replace:: |onedal| K-Means Initialization
.. _onedal-dg-k-means-initialization: https://uxlfoundation.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html#initialization

.. |onedal-dg-qr-decomposition-without-pivoting| replace:: |onedal| QR Decomposition without pivoting
.. _onedal-dg-qr-decomposition-without-pivoting: https://uxlfoundation.github.io/oneDAL/daal/algorithms/qr/qr-without-pivoting.html

.. |onedal-dg-mse| replace:: |onedal| MSE
.. _onedal-dg-mse: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/mse.html

.. |onedal-dg-stochastic-average-gradient-descent-saga| replace:: |onedal| Stochastic Average Gradient Descent SAGA
.. _onedal-dg-stochastic-average-gradient-descent-saga: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/stochastic-average-gradient-accelerated-method.html

.. |onedal-dg-coordinate-descent| replace:: |onedal| Coordinate Descent Algorithm
.. _onedal-dg-coordinate-descent: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/solvers/coordinate-descent.html

.. |onedal-dg-precomputed-objective-function| replace:: |onedal| Objective Function with Precomputed Characteristics
.. _onedal-dg-precomputed-objective-function: https://uxlfoundation.github.io/oneDAL/daal/algorithms/optimization-solvers/objective-functions/with-precomputed-characteristics.html

.. |onedal-dg-engines| replace:: |onedal| Engines
.. _onedal-dg-engines: https://uxlfoundation.github.io/oneDAL/daal/algorithms/engines/index.html

.. |onedal-dg-cholesky-decomposition| replace:: |onedal| Cholesky Decomposition
.. _onedal-dg-cholesky-decomposition: https://uxlfoundation.github.io/oneDAL/daal/algorithms/cholesky/cholesky.html

.. |onedal-dg-classification-decision-forest| replace:: |onedal| Classification Decision Forest
.. _onedal-dg-classification-decision-forest: https://uxlfoundation.github.io/oneDAL/daal/algorithms/decision_forest/decision-forest-classification.html

.. |onedal-dg-normalization| replace:: |onedal| Normalization
.. _onedal-dg-normalization: https://uxlfoundation.github.io/oneDAL/daal/algorithms/normalization/index.html

.. |onedal-dg-density-based-spatial-clustering-of-applications-with-noise| replace:: |onedal| Density-Based Spatial Clustering of Applications with Noise
.. _onedal-dg-density-based-spatial-clustering-of-applications-with-noise: https://uxlfoundation.github.io/oneDAL/daal/algorithms/dbscan/index.html

.. |onedal-dg-moments-of-low-order| replace:: |onedal| Moments of Low Order
.. _onedal-dg-moments-of-low-order: https://uxlfoundation.github.io/oneDAL/daal/algorithms/moments/moments-of-low-order.html

.. |onedal-dg-mt2203| replace:: |onedal| mt2203
.. _onedal-dg-mt2203: https://uxlfoundation.github.io/oneDAL/daal/algorithms/engines/mt2203.html

.. |onedal-dg-normal-distribution| replace:: |onedal| Normal Distribution
.. _onedal-dg-normal-distribution: https://uxlfoundation.github.io/oneDAL/daal/algorithms/distributions/normal.html

.. |onedal-dg-k-means-computation| replace:: |onedal| K-Means Computation
.. _onedal-dg-k-means-computation: https://uxlfoundation.github.io/oneDAL/daal/algorithms/kmeans/k-means-clustering.html#computation

.. |onedal-dg-min-max| replace:: |onedal| Min-Max
.. _onedal-dg-min-max: https://uxlfoundation.github.io/oneDAL/daal/algorithms/normalization/min-max.html
