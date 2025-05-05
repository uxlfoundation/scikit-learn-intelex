# ==============================================================================
# Copyright 2014 Intel Corporation
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
# ==============================================================================

import sys
import warnings
from functools import lru_cache

import sklearn.cluster as cluster_module
import sklearn.decomposition as decomposition_module
import sklearn.ensemble as ensemble_module
import sklearn.linear_model as linear_model_module
import sklearn.linear_model._logistic as logistic_module
import sklearn.manifold as manifold_module
import sklearn.neighbors as neighbors_module
import sklearn.svm as svm_module
from sklearn import metrics, model_selection
from sklearn.utils import validation

from daal4py.sklearn._utils import set_idp_sklearn_verbose

from ..cluster.dbscan import DBSCAN as DBSCAN_daal4py
from ..cluster.k_means import KMeans as KMeans_daal4py
from ..decomposition._pca import PCA as PCA_daal4py
from ..ensemble._forest import RandomForestClassifier as RandomForestClassifier_daal4py
from ..ensemble._forest import RandomForestRegressor as RandomForestRegressor_daal4py
from ..linear_model.coordinate_descent import ElasticNet as ElasticNet_daal4py
from ..linear_model.coordinate_descent import Lasso as Lasso_daal4py
from ..linear_model.linear import LinearRegression as LinearRegression_daal4py
from ..linear_model.logistic_path import LogisticRegression as LogisticRegression_daal4py
from ..linear_model.logistic_path import (
    logistic_regression_path as daal_optimized_logistic_path,
)
from ..linear_model.ridge import Ridge as Ridge_daal4py
from ..manifold import TSNE as TSNE_daal4py
from ..metrics import pairwise_distances, roc_auc_score
from ..model_selection import train_test_split
from ..neighbors import KNeighborsClassifier as KNeighborsClassifier_daal4py
from ..neighbors import KNeighborsRegressor as KNeighborsRegressor_daal4py
from ..neighbors import NearestNeighbors as NearestNeighbors_daal4py
from ..svm.svm import SVC as SVC_daal4py
from ..utils.validation import _assert_all_finite


@lru_cache(maxsize=None)
def _get_map_of_algorithms():
    mapping = {
        "pca": [[(decomposition_module, "PCA", PCA_daal4py), None]],
        "kmeans": [[(cluster_module, "KMeans", KMeans_daal4py), None]],
        "dbscan": [[(cluster_module, "DBSCAN", DBSCAN_daal4py), None]],
        "distances": [[(metrics, "pairwise_distances", pairwise_distances), None]],
        "linear": [
            [(linear_model_module, "LinearRegression", LinearRegression_daal4py), None]
        ],
        "ridge": [[(linear_model_module, "Ridge", Ridge_daal4py), None]],
        "elasticnet": [[(linear_model_module, "ElasticNet", ElasticNet_daal4py), None]],
        "lasso": [[(linear_model_module, "Lasso", Lasso_daal4py), None]],
        "svm": [[(svm_module, "SVC", SVC_daal4py), None]],
        "logistic": [
            [
                (
                    logistic_module,
                    "_logistic_regression_path",
                    daal_optimized_logistic_path,
                ),
                None,
            ]
        ],
        "log_reg": [
            [
                (linear_model_module, "LogisticRegression", LogisticRegression_daal4py),
                None,
            ]
        ],
        "knn_classifier": [
            [
                (neighbors_module, "KNeighborsClassifier", KNeighborsClassifier_daal4py),
                None,
            ]
        ],
        "nearest_neighbors": [
            [(neighbors_module, "NearestNeighbors", NearestNeighbors_daal4py), None]
        ],
        "knn_regressor": [
            [(neighbors_module, "KNeighborsRegressor", KNeighborsRegressor_daal4py), None]
        ],
        "random_forest_classifier": [
            [
                (
                    ensemble_module,
                    "RandomForestClassifier",
                    RandomForestClassifier_daal4py,
                ),
                None,
            ]
        ],
        "random_forest_regressor": [
            [
                (ensemble_module, "RandomForestRegressor", RandomForestRegressor_daal4py),
                None,
            ]
        ],
        "train_test_split": [
            [(model_selection, "train_test_split", train_test_split), None]
        ],
        "fin_check": [[(validation, "_assert_all_finite", _assert_all_finite), None]],
        "roc_auc_score": [[(metrics, "roc_auc_score", roc_auc_score), None]],
        "tsne": [[(manifold_module, "TSNE", TSNE_daal4py), None]],
    }
    mapping["svc"] = mapping["svm"]
    mapping["logisticregression"] = mapping["log_reg"]
    mapping["kneighborsclassifier"] = mapping["knn_classifier"]
    mapping["nearestneighbors"] = mapping["nearest_neighbors"]
    mapping["kneighborsregressor"] = mapping["knn_regressor"]
    mapping["randomforestclassifier"] = mapping["random_forest_classifier"]
    mapping["randomforestregressor"] = mapping["random_forest_regressor"]
    mapping["linearregression"] = mapping["linear"]
    mapping["logisticregression"] = mapping["log_reg"]
    mapping["_logistic_regression_path"] = mapping["logistic"]
    mapping["_assert_all_finite"] = mapping["fin_check"]
    mapping["pairwise_distances"] = mapping["distances"]
    return mapping


def do_patch(name, get_map=_get_map_of_algorithms):
    lname = name.lower()
    if lname in get_map():
        for descriptor in get_map()[lname]:
            which, what, replacer = descriptor[0]
            if descriptor[1] is None:
                descriptor[1] = getattr(which, what, False)
            setattr(which, what, replacer)
    else:
        raise ValueError("Has no patch for: " + name)


def do_unpatch(name, get_map=_get_map_of_algorithms):
    lname = name.lower()
    if lname in get_map():
        for descriptor in get_map()[lname]:
            if descriptor[1] is not None:
                which, what, _ = descriptor[0]
                if descriptor[1]:
                    setattr(which, what, descriptor[1])
                elif hasattr(which, what):
                    delattr(which, what)
    else:
        raise ValueError("Has no patch for: " + name)


def enable(name=None, verbose=True, deprecation=True, get_map=_get_map_of_algorithms):
    if name is not None:
        do_patch(name, get_map)
    else:
        for key in get_map():
            do_patch(key, get_map)
    if deprecation:
        set_idp_sklearn_verbose()
        warnings.warn_explicit(
            "\nScikit-learn patching with daal4py is deprecated "
            "and will be removed in the future.\n"
            "Use Extension "
            "for Scikit-learn* module instead "
            "(pip install scikit-learn-intelex).\n"
            "To enable patching, please use one of the "
            "following options:\n"
            "1) From the command line:\n"
            "    python -m sklearnex <your_script>\n"
            "2) From your script:\n"
            "    from sklearnex import patch_sklearn\n"
            "    patch_sklearn()",
            FutureWarning,
            "dispatcher.py",
            151,
        )
    if verbose and deprecation and sys.stderr is not None:
        sys.stderr.write(
            "oneAPI Data Analytics Library solvers for sklearn enabled: "
            "https://uxlfoundation.github.io/scikit-learn-intelex/\n"
        )


def disable(name=None, get_map=_get_map_of_algorithms):
    if name is not None:
        do_unpatch(name, get_map)
    else:
        for key in get_map():
            do_unpatch(key, get_map)
        get_map.cache_clear()


def _is_enabled(name, get_map=_get_map_of_algorithms):
    lname = name.lower()
    if lname in get_map():
        enabled = True
        for descriptor in get_map()[lname]:
            which, what, replacer = descriptor[0]
            enabled = enabled and getattr(which, what, None) == replacer
        return enabled
    else:
        raise ValueError("Has no patch for: " + name)


def patch_is_enabled(name=None, get_map=_get_map_of_algorithms, return_map=False):
    if name is not None:
        return _is_enabled(name, get_map)
    else:
        if return_map:
            enabled = {}
            for key in get_map():
                enabled[key] = _is_enabled(key, get_map)
        else:
            enabled = True
            for key in get_map():
                enabled = enabled and _is_enabled(key, get_map)
        return enabled


def _patch_names():
    return list(_get_map_of_algorithms().keys())
