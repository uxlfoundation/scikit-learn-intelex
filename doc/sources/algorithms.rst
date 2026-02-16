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

.. include:: substitutions.rst
.. _sklearn_algorithms:

####################
Supported Algorithms
####################

.. note::
   To verify that oneDAL is being used for these algorithms, you can enable verbose mode. 
   See :doc:`verbose` for details.

Applying |sklearnex| impacts the following |sklearn| estimators:

on CPU
------

Classification
**************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.svm.SVC`
     - ``kernel`` must be one of [``"linear"``, ``"rbf"``, ``"poly"``, ``"sigmoid"``]
     - Negative weights are not supported.
     - ``probability=True`` is not supported with array API classes other than NumPy.
   * - :obj:`sklearn.svm.NuSVC`
     - ``kernel`` must be one of [``"linear"``, ``"rbf"``, ``"poly"``, ``"sigmoid"``]
     - Negative weights are not supported.
     - 
   * - :obj:`sklearn.ensemble.RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.ensemble.ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.neighbors.KNeighborsClassifier`
     -
       - For ``algorithm`` == `'kd_tree'`:

         all parameters except ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - For ``algorithm`` == `'brute'`:

         all parameters except ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.linear_model.LogisticRegression`
     - All parameters are supported except:

       - ``solver`` not in [``'lbfgs'``, ``'newton-cg'``]
       - ``l1_ratio`` != ``0``
       - ``dual`` = ``True``
       - ``sample_weight`` != ``None``
       - ``class_weight`` != ``None``
       - Solver ``'newton-cg'`` with ``fit_intercept`` = ``False`` is not supported
     - Sparse data is not supported.
     - Solver ``'newton-cg'`` is **only** available in :doc:`preview mode <preview>`.
   * - :obj:`sklearn.linear_model.LogisticRegressionCV`
     - All parameters are supported except:

       - ``solver`` not in [``'lbfgs'``, ``'newton-cg'``]
       - ``l1_ratios`` not in [``0``, ``"warn"``]
       - ``dual`` = ``True``
       - ``sample_weight`` != ``None``
       - ``class_weight`` != ``None``
       - Solver ``'newton-cg'`` with ``fit_intercept`` = ``False`` is not supported
     - Sparse data is not supported.
     - Estimator is **only** available in :doc:`preview mode <preview>`.

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.svm.SVR`
     - ``kernel`` must be one of [``"linear"``, ``"rbf"``, ``"poly"``, ``"sigmoid"``]
     - Negative weights are not supported.
   * - :obj:`sklearn.svm.NuSVR`
     - ``kernel`` must be one of [``"linear"``, ``"rbf"``, ``"poly"``, ``"sigmoid"``]
     - Negative weights are not supported.
   * - :obj:`sklearn.ensemble.RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.ensemble.ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.neighbors.KNeighborsRegressor`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.linear_model.LinearRegression`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
       - ``positive`` = `True` (this is supported through the class :obj:`sklearn.linear_model.ElasticNet`)
     - Only dense data is supported.
   * - :obj:`sklearn.linear_model.Ridge`
     - All parameters are supported except:

       - ``solver`` != `'auto'`
       - ``sample_weight`` != `None`
       - ``positive`` = `True` (this is supported through the class :obj:`sklearn.linear_model.ElasticNet`)
       - ``alpha`` must be a scalar
     - Only dense data is supported.
   * - :obj:`sklearn.linear_model.ElasticNet`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
     - Sparse data is not supported.
   * - :obj:`sklearn.linear_model.Lasso`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
     - Sparse data is not supported.

Clustering
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.cluster.KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``verbose`` = ``True`` will only print results from the last iteration, and will only print
         inertia numbers, not 'convergence achieved' messages.
     - No limitations
   * - :obj:`sklearn.cluster.DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.decomposition.PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - 
   * - :obj:`sklearn.decomposition.IncrementalPCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.
   * - :obj:`sklearn.manifold.TSNE`
     - All parameters are supported except:

       - ``metric`` != 'euclidean' or `'minkowski'` with ``p`` != `2`
       - ``n_components`` can only be `2`
       - ``method`` != ``"barnes_hut"``
       
       Refer to :ref:`TSNE acceleration details <acceleration_tsne>` to learn more.
     - Sparse data is not supported for the initialization and distance calculation stages.
     - 

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.neighbors.NearestNeighbors`
     -
       - For ``algorithm`` == 'kd_tree':

         all parameters except ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - For ``algorithm`` == 'brute':

         all parameters except ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Sparse data is not supported

Other Tasks
***********

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.covariance.EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.
   * - :obj:`sklearnex.basic_statistics.BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
     - Sample weights are **not** supported for CSR data format
   * - :obj:`sklearn.model_selection.train_test_split`
     - All parameters are supported
     - Supported data formats:
        
       - Only dense data is supported
       - Only integer and 32/64-bits floating point types are supported
       - Data with more than 3 dimensions is not supported
       - Only ``np.ndarray``, ``pd.DataFrame`` and ``pd.Series`` inputs are supported.
     - 
   * - :obj:`sklearn.utils.assert_all_finite`
     - All parameters are supported
     - Only dense data is supported
     - 
   * - :obj:`sklearn.metrics.pairwise_distance`
     - All parameters are supported except:

       - ``metric`` not in [`'cosine'`, `'correlation'`]
     - Supported data formats:

       - Only dense data is supported
       - ``Y`` must be `None`
       - Input dtype must be `np.float64`
     - 
   * - :obj:`sklearn.metrics.roc_auc_score`
     - All parameters are supported except:

       - ``average`` != `None`
       - ``sample_weight`` != `None`
       - ``max_fpr`` != `None`
       - ``multi_class`` != `None`
     - No limitations
     - 

on GPU
------
.. _sklearn_algorithms_gpu:

.. seealso:: :ref:`oneapi_gpu`

Classification
**************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.svm.SVC`
     - All parameters are supported except:

       - ``kernel`` not in [``"linear"``, ``"rbf"``]
       - ``class_weight`` != ``None``
       - ``probability`` != ``False`` (see limitations on this)
     - Only dense data is supported. Negative weights are not supported.
     - Only binary classification is supported. If passing ``probability=True``,
       under :ref:`target_offload <target_offload>`, the initial model will be
       fitted on GPU, but subsequent calibration will be done on CPU through |sklearn|;
       whereas if passing array API inputs with ``probability=True``, the whole
       procedure will fall back to |sklearn| on CPU.
   * - :obj:`sklearn.ensemble.RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.ensemble.ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.neighbors.KNeighborsClassifier`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.linear_model.LogisticRegression`
     - All parameters are supported except:

       - ``solver`` != `'newton-cg'`
       - ``class_weight`` != `None`
       - ``sample_weight`` != `None`
       - ``dual`` = ``True``
       - ``intercept_scaling`` != `1`
       - ``warm_start`` = ``True``
       - ``l1_ratio`` != ``0``
     - No limitations.
     - Only binary classification is supported.

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.ensemble.RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.ensemble.ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.neighbors.KNeighborsRegressor`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Only dense data is supported
   * - :obj:`sklearn.linear_model.Ridge`
     - All parameters are supported except:

       - ``solver`` != `'auto'`
       - ``sample_weight`` != `None`
       - ``positive`` = `True`
       - ``alpha`` must be a scalar
     - Only dense data is supported.

   * - :obj:`sklearn.linear_model.LinearRegression`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
       - ``positive`` = `True`
     - Only dense data is supported.

Clustering
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.cluster.KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``init`` = ``'k-means++'`` falls back to CPU
       - ``verbose`` = ``True`` will only print results from the last iteration, and will only print
         inertia numbers, not 'convergence achieved' messages.
     - No limitations
   * - :obj:`sklearn.cluster.DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.decomposition.PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - 
   * - :obj:`sklearn.decomposition.IncrementalPCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.neighbors.NearestNeighbors`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported

Other Tasks
***********

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.covariance.EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.
   * - :obj:`sklearnex.basic_statistics.BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
     - Sample weights are **not** supported for CSR data format.

.. _spmd-support:

SPMD Support
------------

.. seealso:: :ref:`distributed`

Classification
**************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
     - Other limitations
   * - :obj:`sklearn.ensemble.RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.ensemble.ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.neighbors.KNeighborsClassifier`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
       - ``predict_proba`` method not supported
     - Only dense data is supported.
     - Number of classes must be at least 2.
   * - :obj:`sklearn.linear_model.LogisticRegression`
     - All parameters are supported except:

       - ``solver`` != `'newton-cg'`
       - ``class_weight`` != `None`
       - ``sample_weight`` != `None`
       - ``dual`` = ``True``
       - ``intercept_scaling`` != `1`
       - ``warm_start`` = ``True``
       - ``l1_ratio`` != ``0``
     - No limitations
     - Only binary classification is supported

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
   * - :obj:`sklearn.ensemble.RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.ensemble.ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != ``'squared_error'``
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
       - ``n_estimators`` > ``6024``
     - Multi-output and sparse data are not supported
   * - :obj:`sklearn.neighbors.KNeighborsRegressor`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Only dense data is supported
   * - :obj:`sklearn.linear_model.LinearRegression`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
       - ``positive`` = `True`
     - Only dense data is supported.

Clustering
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
   * - :obj:`sklearn.cluster.KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``init`` = `'k-means++'` falls back to CPU
       - ``verbose`` = ``True`` will only print results from the last iteration, and will only print
         inertia numbers, not 'convergence achieved' messages.
     - No limitations
   * - :obj:`sklearn.cluster.DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
     - Other limitations
   * - :obj:`sklearn.decomposition.PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - 
   * - :obj:`sklearn.decomposition.IncrementalPCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - :obj:`sklearn.neighbors.NearestNeighbors`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported

Other Tasks
***********

.. list-table::
   :widths: 10 40 10 10
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
     - Other limitations
   * - :obj:`sklearn.covariance.EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
     - Estimator is **only** available in :doc:`preview mode <preview>`.
   * - :obj:`sklearnex.basic_statistics.BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
     - Sample weights **not** supported for CSR data format

Scikit-learn Tests
------------------

Monkey-patched scikit-learn classes and functions passes scikit-learn's own test
suite, with few exceptions - see :ref:`conformance_tests` for details.
