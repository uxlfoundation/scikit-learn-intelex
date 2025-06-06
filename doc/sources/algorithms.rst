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
   See :ref:`verbose mode documentation <verbose>` for details.

Applying |sklearnex| impacts the following |sklearn| estimators:

on CPU
------

Classification
**************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `SVC`
     - All parameters are supported
     - No limitations
   * - `NuSVC`
     - All parameters are supported
     - No limitations
   * - `RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
     - Multi-output and sparse data are not supported
   * - `KNeighborsClassifier`
     -
       - For ``algorithm`` == `'kd_tree'`:

         all parameters except ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - For ``algorithm`` == `'brute'`:

         all parameters except ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Multi-output and sparse data are not supported
   * - `LogisticRegression`
     - All parameters are supported except:

       - ``solver`` != `'newton-cg'`
       - ``class_weight`` != `None`
       - ``sample_weight`` != `None`
       - ``penalty`` != `'l2'`
       - ``dual`` = `True`
       - ``intercept_scaling`` != `1`
       - ``multi_class`` != `'multinomial'`
       - ``warm_start`` = `True`
       - ``l1_ratio`` != `None`
       - Only binary classification is supported
     - Supported data formats:
         - Dense data
         - CSR sparse matrices

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `SVR`
     - All parameters are supported
     - No limitations
   * - `NuSVR`
     - All parameters are supported
     - No limitations
   * - `RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
     - Multi-output and sparse data are not supported
   * - `KNeighborsRegressor`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Multi-output and sparse data are not supported
   * - `LinearRegression`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
       - ``positive`` = `True`
     - Only dense data is supported.
   * - `Ridge`
     - All parameters are supported except:

       - ``solver`` != `'auto'`
       - ``sample_weight`` != `None`
       - ``positive`` = `True`
       - ``alpha`` must be scalar
     - Only dense data is supported, `#observations` should be >= `#features`.
   * - `ElasticNet`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported, `#observations` should be >= `#features`.
   * - `Lasso`
     - All parameters are supported except:

       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported, `#observations` should be >= `#features`.

Clustering
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``init`` = `'k-means++'` fallbacks to CPU
     - Supported data formats:
         - Dense data
         - CSR sparse matrices
   * - `DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported
   * - `TSNE`
     - All parameters are supported except:

       - ``metric`` != 'euclidean' or `'minkowski'` with ``p`` != `2`

       - ``n_components`` can only be `2`

       Refer to :ref:`TSNE acceleration details <acceleration_tsne>` to learn more.
     - Sparse data is not supported

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `NearestNeighbors`
     -
       - For ``algorithm`` == 'kd_tree':

         all parameters except ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
       - For ``algorithm`` == 'brute':

         all parameters except ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Sparse data is not supported

Other Tasks
***********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
   * - `BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
       - Sample weights **not** supported for CSR data format
   * - `train_test_split`
     - All parameters are supported
     - Supported data formats:
        
       - Only dense data is supported
       - Only integer and 32/64-bits floating point types are supported
       - Data with more than 3 dimensions is not supported
       - The input has to be a np.ndarray object
   * - `assert_all_finite`
     - All parameters are supported
     - Only dense data is supported
   * - `pairwise_distance`
     - All parameters are supported except:

       - ``metric`` not in [`'cosine'`, `'correlation'`]
     - Supported data formats:

       - Only dense data is supported
       - ``Y`` must be `None`
       - Input dtype must be `np.float64`
   * - `roc_auc_score`
     - All parameters are supported except:

       - ``average`` != `None`
       - ``sample_weight`` != `None`
       - ``max_fpr`` != `None`
       - ``multi_class`` != `None`
     - No limitations

on GPU
------

.. seealso:: :ref:`oneapi_gpu`

Classification
**************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `SVC`
     - All parameters are supported except:

       - ``kernel`` = `'sigmoid_poly'`
       - ``class_weight`` != `None`
     - Only binary dense data is supported
   * - `RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `KNeighborsClassifier`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported
   * - `LogisticRegression`
     - All parameters are supported except:

       - ``solver`` != `'newton-cg'`
       - ``class_weight`` != `None`
       - ``sample_weight`` != `None`
       - ``penalty`` != `'l2'`
       - ``dual`` = `True`
       - ``intercept_scaling`` != `1`
       - ``multi_class`` != `'multinomial'`
       - ``warm_start`` = `True`
       - ``l1_ratio`` != `None`
       - Only binary classification is supported
     - Supported data formats:
         - Dense data
         - CSR sparse matrices

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `KNeighborsRegressor`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Only dense data is supported
   * - `LinearRegression`
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
   * - `KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``init`` = `'k-means++'` fallbacks to CPU
     - Supported data formats:
         - Dense data
         - CSR sparse matrices
   * - `DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `NearestNeighbors`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported

Other Tasks
***********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
   * - `BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
       - Sample weights **not** supported for CSR data format

.. _spmd-support:

SPMD Support
------------

.. seealso:: :ref:`distributed`

Classification
**************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
   * - `RandomForestClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesClassifier`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'gini'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `KNeighborsClassifier`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
       - ``predict_proba`` method not supported
     - Only dense data is supported
   * - `LogisticRegression`
     - All parameters are supported except:

       - ``solver`` != `'newton-cg'`
       - ``class_weight`` != `None`
       - ``sample_weight`` != `None`
       - ``penalty`` != `'l2'`
       - ``dual`` = `True`
       - ``intercept_scaling`` != `1`
       - ``multi_class`` != `'multinomial'`
       - ``warm_start`` = `True`
       - ``l1_ratio`` != `None`
       - Only binary classification is supported
     - Supported data formats:
         - Dense data
         - CSR sparse matrices

Regression
**********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
   * - `RandomForestRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `ExtraTreesRegressor`
     - All parameters are supported except:

       - ``warm_start`` = `True`
       - ``ccp_alpha`` != `0`
       - ``criterion`` != `'mse'`
       - ``oob_score`` = `True`
       - ``sample_weight`` != `None`
     - Multi-output and sparse data are not supported
   * - `KNeighborsRegressor`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
     - Only dense data is supported
   * - `LinearRegression`
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
   * - `KMeans`
     - All parameters are supported except:

       - ``algorithm`` != ``'lloyd'`` ('elkan' falls back to 'lloyd')
       - ``n_clusters`` = ``1``
       - ``sample_weight`` must be None, constant, or equal weights
       - ``init`` = `'k-means++'` fallbacks to CPU
     - Supported data formats:
         - Dense data
         - CSR sparse matrices
   * - `DBSCAN`
     - All parameters are supported except:

       - ``metric`` != `'euclidean'`
       - ``algorithm`` not in [`'brute'`, `'auto'`]
     - Only dense data is supported

Dimensionality Reduction
************************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters & Methods
     - Data formats
   * - `PCA`
     - All parameters are supported except:

       - ``svd_solver`` not in [`'full'`, `'covariance_eigh'`, `'onedal_svd'`]
       - For |sklearn| < 1.5: `'full'` solver is automatically mapped to `'covariance_eigh'`
     - Sparse data is not supported

Nearest Neighbors
*****************

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `NearestNeighbors`
     - All parameters are supported except:

       - ``algorithm`` != `'brute'`
       - ``weights`` = `'callable'`
       - ``metric`` not in [`'euclidean'`, `'manhattan'`, `'minkowski'`, `'chebyshev'`, `'cosine'`]
     - Only dense data is supported

Other Tasks
***********

.. list-table::
   :widths: 10 30 20
   :header-rows: 1
   :align: left

   * - Algorithm
     - Parameters
     - Data formats
   * - `EmpiricalCovariance`
     - All parameters are supported
     - Only dense data is supported
   * - `BasicStatistics`
     - All parameters are supported
     - Supported data formats:

       - Dense data
       - CSR sparse matrices
       - Sample weights **not** supported for CSR data format

Scikit-learn Tests
------------------

Monkey-patched scikit-learn classes and functions passes scikit-learn's own test
suite, with few exceptions, specified in `deselected_tests.yaml
<https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/deselected_tests.yaml>`__.
