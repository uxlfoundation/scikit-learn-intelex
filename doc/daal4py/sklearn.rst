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

.. _sklearn:

#############################
Scikit-Learn API and patching
#############################


Python interface to efficient |onedal| provided by daal4py allows one
to create scikit-learn compatible estimators, transformers, clusterers, etc. powered by oneDAL which
are nearly as efficient as native programs.

Deprecation Notice
-------------------------------

Scikit-learn patching functionality in daal4py was deprecated and moved to a separate
package, `Extension for Scikit-learn* <https://github.com/uxlfoundation/scikit-learn-intelex>`_.
All future patches will be available only in Extension for Scikit-learn*.
Please use the scikit-learn-intelex package instead of daal4py for the scikit-learn acceleration.

.. _sklearn_patches:

oneDAL accelerated scikit-learn
-------------------------------

daal4py can dynamically patch scikit-learn estimators to use |onedal|
as the underlying solver, while getting the same solution faster.

It is possible to enable those patches without editing the code of a scikit-learn application by
using the following commandline flag::

    python -m daal4py my_application.py

If you are using Scikit-Learn from Intel® Distribution for Python, then
you can enable daal4py patches through an environment variable. To do this, set ``USE_DAAL4PY_SKLEARN`` to one of the values
``True``, ``'1'``, ``'y'``, ``'yes'``, ``'Y'``, ``'YES'``, ``'Yes'``, ``'true'``, ``'True'`` or ``'TRUE'`` as shown below.

On Linux and Mac OS::

    export USE_DAAL4PY_SKLEARN=1

On Windows::

    set USE_DAAL4PY_SKLEARN=1

To disable daal4py patches, set the ``USE_DAAL4PY_SKLEARN`` environment variable to 0.

Patches can also be enabled programmatically::

    import daal4py.sklearn
    daal4py.sklearn.patch_sklearn()

It is possible to undo the patch with::

    daal4py.sklearn.unpatch_sklearn()

.. _sklearn_algorithms:

Applying the monkey patch will impact the following existing scikit-learn
algorithms:

.. list-table::
   :widths: 10 10 30 15
   :header-rows: 1
   :align: left

   * - Task
     - Functionality
     - Parameters support
     - Data support
   * - Classification
     - SVC
     - All parameters except ``poly`` and ``sigmoid`` kernels.
     - No limitations.
   * - Classification
     - RandomForestClassifier
     - All parameters except ``warm_start`` = True, ``ccp_alpha`` != 0, ``criterion`` != 'gini', ``oob_score`` = True.
     - Multi-output, sparse data and out-of-bag score are not supported.
   * - Classification
     - KNeighborsClassifier
     - All parameters except ``metric`` != 'euclidean' or ``minkowski`` with ``p`` = 2.
     - Multi-output and sparse data is not supported.
   * - Classification
     - LogisticRegression
     - All parameters except ``solver`` != 'lbfgs' or 'newton-cg', ``class_weight`` != None, ``sample_weight`` != None.
     - Only dense data is supported.
   * - Regression
     - RandomForestRegressor
     - All parameters except ``warm_start`` = True, ``ccp_alpha`` != 0, ``criterion`` != 'mse', ``oob_score`` = True.
     - Multi-output, sparse data and out-of-bag score are not supported.
   * - Regression
     - KNeighborsRegressor
     - All parameters except ``metric`` != 'euclidean' or ``minkowski`` with ``p`` = 2.
     - Multi-output and sparse data is not supported.
   * - Regression
     - LinearRegression
     - All parameters except ``normalize`` != False and ``sample_weight`` != None.
     - Only dense data is supported, #observations should be >= #features.
   * - Regression
     - Ridge
     - All parameters except ``normalize`` != False, ``solver`` != 'auto' and ``sample_weight`` != None.
     - Only dense data is supported, #observations should be >= #features.
   * - Regression
     - ElasticNet
     - All parameters except ``sample_weight`` != None.
     - Multi-output and sparse data is not supported, #observations should be >= #features.
   * - Regression
     - Lasso
     - All parameters except ``sample_weight`` != None.
     - Multi-output and sparse data is not supported, #observations should be >= #features.
   * - Clustering
     - KMeans
     - All parameters except ``precompute_distances`` and ``sample_weight`` != None.
     - No limitations.
   * - Clustering
     - DBSCAN
     - All parameters except ``metric`` != 'euclidean' or ``minkowski`` with ``p`` = 2.
     - Only dense data is supported.
   * - Dimensionality reduction
     - PCA
     - All parameters except ``svd_solver`` != 'full'.
     - Sparse data is not supported.
   * - Unsupervised
     - NearestNeighbors
     - All parameters except ``metric`` != 'euclidean' or ``minkowski`` with ``p`` = 2.
     - Sparse data is not supported.
   * - Other
     - train_test_split
     - All parameters are supported.
     - Only dense data is supported.
   * - Other
     - assert_all_finite
     - All parameters are supported.
     - Only dense data is supported.
   * - Other
     - pairwise_distance
     - With metric=``cosine`` and ``correlation``.
     - Only dense data is supported.
   * - Other
     - roc_auc_score
     - Parameters ``average``, ``sample_weight``, ``max_fpr`` and ``multi_class`` are not supported.
     - No limitations.


Monkey-patched scikit-learn classes and functions passes scikit-learn's own test
suite, with few exceptions, specified in `deselected_tests.yaml
<https://github.com/IntelPython/daal4py/blob/main/deselected_tests.yaml>`__.

In particular the tests execute `check_estimator
<https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>`__
on all added and monkey-patched classes, which are discovered by means of
introspection. This assures scikit-learn API compatibility of all
`daal4py.sklearn` classes.

.. note::
    daal4py supports optimizations for the last four versions of scikit-learn.
    The latest release of daal4py-2021.1 supports scikit-learn 0.21.X, 0.22.X, 0.23.X and 0.24.X.

.. _sklearn_verbose:

scikit-learn verbose
--------------------

To find out which implementation of the algorithm is currently used,
set the environment variable.

On Linux and Mac OS::

    export IDP_SKLEARN_VERBOSE=INFO

On Windows::

    set IDP_SKLEARN_VERBOSE=INFO

During the calls that use Intel-optimized scikit-learn, you will receive additional print statements
that indicate which implementation is being called.
These print statements are only available for :ref:`scikit-learn algorithms with daal4py patches <sklearn_algorithms>`.

For example, for DBSCAN you get one of these print statements depending on which implementation is used::

    INFO: sklearn.cluster.DBSCAN.fit: running accelerated version on CPU

::

    INFO: sklearn.cluster.DBSCAN.fit: fallback to original Scikit-learn


.. _sklearn_api:

scikit-learn API
----------------

The ``daal4py.sklearn`` package contains scikit-learn compatible API which
implement a subset of scikit-learn algorithms using |onedal|.

Currently, these include:

1. ``daal4py.sklearn.neighbors.KNeighborsClassifier``
2. ``daal4py.sklearn.neighbors.KNeighborsRegressor``
3. ``daal4py.sklearn.neighbors.NearestNeighbors``
4. ``daal4py.sklearn.tree.DecisionTreeClassifier``
5. ``daal4py.sklearn.ensemble.RandomForestClassifier``
6. ``daal4py.sklearn.ensemble.RandomForestRegressor``
7. ``daal4py.sklearn.ensemble.AdaBoostClassifier``
8. ``daal4py.sklearn.cluster.KMeans``
9. ``daal4py.sklearn.cluster.DBSCAN``
10. ``daal4py.sklearn.decomposition.PCA``
11. ``daal4py.sklearn.linear_model.Ridge``
12. ``daal4py.sklearn.svm.SVC``
13. ``daal4py.sklearn.linear_model.logistic_regression_path``
14. ``daal4py.sklearn.linear_model.LogisticRegression``
15. ``daal4py.sklearn.linear_model.ElasticNet``
16. ``daal4py.sklearn.linear_model.Lasso``
17. ``daal4py.sklearn.model_selection._daal_train_test_split``
18. ``daal4py.sklearn.metrics._daal_roc_auc_score``

These classes are always available, whether the scikit-learn itself has been
patched, or not. For example::

    import daal4py.sklearn
    daal4py.sklearn.unpatch_sklearn()
    import sklearn.datasets, sklearn.svm

    digits = sklearn.datasets.load_digits()
    X, y = digits.data, digits.target

    clf_d = daal4py.sklearn.svm.SVC(kernel='rbf', gamma='scale', C = 0.5).fit(X, y)
    clf_v = sklearn.svm.SVC(kernel='rbf', gamma='scale', C =0.5).fit(X, y)

    clf_d.score(X, y) # output: 0.9905397885364496
    clf_v.score(X, y) # output: 0.9905397885364496
