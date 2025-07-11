# ===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import random
from collections.abc import Iterable
from functools import partial
from numbers import Number

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)

import daal4py as d4p
from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import _as_numpy, get_dataframes_and_queues
from sklearnex.basic_statistics import BasicStatistics
from sklearnex.cluster import DBSCAN, KMeans
from sklearnex.decomposition import PCA
from sklearnex.metrics import pairwise_distances, roc_auc_score
from sklearnex.model_selection import train_test_split
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearnex.svm import SVC
from sklearnex.tests.utils import (
    _IS_INTEL,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    call_method,
    gen_dataset,
    gen_models_info,
    sklearn_clone_dict,
)

# to reproduce errors even in CI
d4p.daalinit(nthreads=100)

_dataset_dict = {
    "classification": [
        partial(load_iris, return_X_y=True),
        partial(load_breast_cancer, return_X_y=True),
    ],
    "regression": [
        partial(load_diabetes, return_X_y=True),
        partial(
            make_regression, n_samples=500, n_features=10, noise=64.0, random_state=42
        ),
    ],
}


def eval_method(X, y, est, method):
    res = []
    est.fit(X, y)

    if method:
        res = call_method(est, method, X, y)

    if not isinstance(res, Iterable):
        results = [_as_numpy(res)] if res is not est else []
    elif hasattr(res, "dtype") and res.dtype not in [np.object_, np.flexible]:
        # array_api inputs do not have an __iter__ and this also allows for
        # multidimensional array-like inputs to be analyzed without slicing.
        # The check for object and flexible types is because of the
        # radius_neighbors method in kNeighors algorithms which can return
        # numpy object arrays of numpy numeric arrays
        results = [_as_numpy(res).flatten()]
    else:
        results = [_as_numpy(i) for i in res]

    attributes = [method] * len(results)

    # if estimator follows sklearn design rules, then set attributes should have a
    # trailing underscore
    attributes += [
        i
        for i in dir(est)
        if hasattr(est, i) and not i.startswith("_") and i.endswith("_")
    ]
    results += [getattr(est, i) for i in attributes if i != method]
    return results, attributes


def _run_test(estimator, method, datasets):

    for X, y in datasets:
        baseline, attributes = eval_method(X, y, estimator, method)

        for i in range(10):
            res, _ = eval_method(X, y, estimator, method)

            for r, b, n in zip(res, baseline, attributes):
                if (
                    isinstance(b, Number)
                    or hasattr(b, "__array__")
                    or hasattr(b, "__array_namespace__")
                    or hasattr(b, "__sycl_usm_ndarray__")
                ):
                    assert_allclose(
                        r, b, rtol=0.0, atol=0.0, err_msg=str(n + " is incorrect")
                    )


_sparse_instances = [SVC()]
if daal_check_version((2025, "P", 200)):  # Test for >= 2025.2.0
    _sparse_instances.extend(
        [
            BasicStatistics(result_options=["sum", "min"]),
        ]
    )
if daal_check_version((2024, "P", 700)):  # Test for > 2024.7.0
    _sparse_instances.extend(
        [
            KMeans(),
            KMeans(init="random"),
            KMeans(init="k-means++"),
        ]
    )
SPARSE_INSTANCES = sklearn_clone_dict({str(i): i for i in _sparse_instances})

STABILITY_INSTANCES = sklearn_clone_dict(
    {
        str(i): i
        for i in [
            KNeighborsClassifier(algorithm="brute", weights="distance"),
            KNeighborsClassifier(algorithm="kd_tree", weights="distance"),
            KNeighborsClassifier(algorithm="kd_tree"),
            KNeighborsRegressor(algorithm="brute", weights="distance"),
            KNeighborsRegressor(algorithm="kd_tree", weights="distance"),
            KNeighborsRegressor(algorithm="kd_tree"),
            NearestNeighbors(algorithm="kd_tree"),
            DBSCAN(algorithm="brute"),
            PCA(n_components=0.5, svd_solver="covariance_eigh"),
            KMeans(init="random"),
        ]
    }
)


def _skip_neighbors(estimator, method):
    if (
        not _IS_INTEL
        and ("Neighbors" in estimator or "LocalOutlierFactor" in estimator)
        and method
        in ["score", "predict", "kneighbors", "kneighbors_graph", "predict_proba"]
    ):
        pytest.skip(f"{estimator} shows instability on non-Intel(R) hardware")


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy,array_api"))
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_stability(estimator, method, dataframe, queue):
    if estimator in ["LogisticRegression", "TSNE"]:
        pytest.skip(f"stability not guaranteed for {estimator}")
    if estimator in ["KMeans", "PCA"] and "score" in method and queue == None:
        pytest.skip(f"variation observed in {estimator}.score")
    if estimator in ["IncrementalEmpiricalCovariance"] and method == "mahalanobis":
        pytest.skip("allowed fallback to sklearn occurs")
    _skip_neighbors(estimator, method)

    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")

    est = PATCHED_MODELS[estimator]()

    if method and not hasattr(est, method):
        pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(**params)

    datasets = gen_dataset(est, datasets=_dataset_dict, queue=queue, target_df=dataframe)
    _run_test(est, method, datasets)


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy,array_api"))
@pytest.mark.parametrize("estimator, method", gen_models_info(SPECIAL_INSTANCES))
def test_special_estimator_stability(estimator, method, dataframe, queue):
    if queue is None and estimator in ["LogisticRegression(solver='newton-cg')"]:
        pytest.skip(f"stability not guaranteed for {estimator}")
    if "KMeans" in estimator and method == "score" and queue == None:
        pytest.skip(f"variation observed in KMeans.score")
    if estimator == "BasicStatistics()" and queue == None:
        pytest.skip(f"BasicStatistics not deterministic")
    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")
    _skip_neighbors(estimator, method)

    est = SPECIAL_INSTANCES[estimator]

    if method and not hasattr(est, method):
        pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(**params)

    datasets = gen_dataset(est, datasets=_dataset_dict, queue=queue, target_df=dataframe)
    _run_test(est, method, datasets)


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy,array_api"))
@pytest.mark.parametrize("estimator, method", gen_models_info(SPARSE_INSTANCES))
def test_sparse_estimator_stability(estimator, method, dataframe, queue):
    if "KMeans" in estimator and method in "score" and queue == None:
        pytest.skip(f"variation observed in KMeans.{method}")
    if (
        not daal_check_version((2025, "P", 0))
        and "KMeans()" in estimator
        and queue == None
    ):
        pytest.skip(f"variation observed in KMeans.{method} in 2024.7 oneDAL")
    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")
    _skip_neighbors(estimator, method)

    est = SPARSE_INSTANCES[estimator]

    if method and not hasattr(est, method):
        pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(**params)

    datasets = gen_dataset(
        est, sparse=True, datasets=_dataset_dict, queue=queue, target_df=dataframe
    )
    _run_test(est, method, datasets)


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues("numpy,array_api"))
@pytest.mark.parametrize("estimator, method", gen_models_info(STABILITY_INSTANCES))
def test_other_estimator_stability(estimator, method, dataframe, queue):
    if "KMeans" in estimator and method == "score" and queue == None:
        pytest.skip(f"variation observed in KMeans.score")
    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")
    _skip_neighbors(estimator, method)

    est = STABILITY_INSTANCES[estimator]

    if method and not hasattr(est, method):
        pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")

    params = est.get_params().copy()
    if "random_state" in params:
        params["random_state"] = 0
        est.set_params(**params)

    datasets = gen_dataset(est, datasets=_dataset_dict, queue=queue, target_df=dataframe)
    _run_test(est, method, datasets)


@pytest.mark.parametrize("features", range(5, 10))
def test_train_test_split(features):
    X, y = make_classification(
        n_samples=4000,
        n_features=features,
        n_informative=features,
        n_redundant=0,
        n_clusters_per_class=8,
        random_state=0,
    )
    (
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
    ) = train_test_split(X, y, test_size=0.33, random_state=0)
    baseline = [baseline_X_train, baseline_X_test, baseline_y_train, baseline_y_test]
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0
        )
        res = [X_train, X_test, y_train, y_test]
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(
                a, b, rtol=0.0, atol=0.0, err_msg=str("train_test_split is incorrect")
            )


@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_pairwise_distances(metric):
    X = np.random.rand(1000)
    X = np.array(X, dtype=np.float64)
    baseline = pairwise_distances(X.reshape(1, -1), metric=metric)
    for _ in range(5):
        res = pairwise_distances(X.reshape(1, -1), metric=metric)
        for a, b in zip(res, baseline):
            np.testing.assert_allclose(
                a, b, rtol=0.0, atol=0.0, err_msg=str("pairwise_distances is incorrect")
            )


@pytest.mark.parametrize("array_size", [100, 1000, 10000])
def test_roc_auc(array_size):
    a = [random.randint(0, 1) for i in range(array_size)]
    b = [random.randint(0, 1) for i in range(array_size)]
    baseline = roc_auc_score(a, b)
    for _ in range(5):
        res = roc_auc_score(a, b)
        np.testing.assert_allclose(
            baseline, res, rtol=0.0, atol=0.0, err_msg=str("roc_auc is incorrect")
        )
