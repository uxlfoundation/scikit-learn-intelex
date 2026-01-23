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
from types import ModuleType
from typing import Optional, Union

import sklearn.cluster as cluster_module
import sklearn.decomposition as decomposition_module
import sklearn.ensemble as ensemble_module
import sklearn.linear_model as linear_model_module
import sklearn.linear_model._logistic as logistic_module
import sklearn.manifold as manifold_module
import sklearn.neighbors as neighbors_module
import sklearn.svm as svm_module
from sklearn import metrics, model_selection
from sklearn.cluster import DBSCAN as DBSCAN_sklearn
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
from sklearn.linear_model import LogisticRegressionCV as LogisticRegressionCV_sklearn
from sklearn.linear_model import Ridge as Ridge_sklearn
from sklearn.linear_model._logistic import (
    _logistic_regression_path as _logistic_regression_path_sklearn,
)
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import pairwise_distances as pairwise_distances_sklearn
from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
from sklearn.model_selection import train_test_split as train_test_split_sklearn
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearn
from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearn
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_sklearn
from sklearn.svm import SVC as SVC_sklearn
from sklearn.utils import validation
from sklearn.utils.validation import _assert_all_finite as _assert_all_finite_sklearn

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
    LogisticRegressionCV as LogisticRegressionCV_daal4py,
)
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

# dict key: sklearn name
# dict value: tuple entries:
# - module from sklearn
# - name of class/function within module
# - sklearnex/daal4py replacement
# - sklearn original if any
PatchMap = dict[str, tuple[ModuleType, str, object, Optional[object]]]


# Note: the keys of this dict are only used as internal IDs to keep
# track of things, and in the functions to check if a given function
# or class is patched. The keys can be arbitrary strings that do not
# necessarily correspond to module paths, but having the full paths
# and names of what they patch makes them easier to identify and debug.
@lru_cache(maxsize=None)
def _get_map_of_algorithms() -> PatchMap:
    mapping = {
        "sklearn.decomposition.PCA": (
            decomposition_module,
            "PCA",
            PCA_daal4py,
            PCA_sklearn,
        ),
        "sklearn.cluster.KMeans": (
            cluster_module,
            "KMeans",
            KMeans_daal4py,
            KMeans_sklearn,
        ),
        "sklearn.cluster.DBSCAN": (
            cluster_module,
            "DBSCAN",
            DBSCAN_daal4py,
            DBSCAN_sklearn,
        ),
        "sklearn.metrics.pairwise_distances": (
            metrics,
            "pairwise_distances",
            pairwise_distances,
            pairwise_distances_sklearn,
        ),
        "sklearn.linear_model.LinearRegression": (
            linear_model_module,
            "LinearRegression",
            LinearRegression_daal4py,
            LinearRegression_sklearn,
        ),
        "sklearn.linear_model.Ridge": (
            linear_model_module,
            "Ridge",
            Ridge_daal4py,
            Ridge_sklearn,
        ),
        "sklearn.linear_model.ElasticNet": (
            linear_model_module,
            "ElasticNet",
            ElasticNet_daal4py,
            ElasticNet_sklearn,
        ),
        "sklearn.linear_model.Lasso": (
            linear_model_module,
            "Lasso",
            Lasso_daal4py,
            Lasso_sklearn,
        ),
        "sklearn.svm.SVC": (svm_module, "SVC", SVC_daal4py, SVC_sklearn),
        "sklearn.linear_model._logistic._logistic_regression_path": (
            logistic_module,
            "_logistic_regression_path",
            daal_optimized_logistic_path,
            _logistic_regression_path_sklearn,
        ),
        "sklearn.linear_model.LogisticRegression": (
            linear_model_module,
            "LogisticRegression",
            LogisticRegression_daal4py,
            LogisticRegression_sklearn,
        ),
        "sklearn.linear_model.LogisticRegressionCV": (
            linear_model_module,
            "LogisticRegressionCV",
            LogisticRegressionCV_daal4py,
            LogisticRegressionCV_sklearn,
        ),
        "sklearn.neighbors.KNeighborsClassifier": (
            neighbors_module,
            "KNeighborsClassifier",
            KNeighborsClassifier_daal4py,
            KNeighborsClassifier_sklearn,
        ),
        "sklearn.neighbors.NearestNeighbors": (
            neighbors_module,
            "NearestNeighbors",
            NearestNeighbors_daal4py,
            NearestNeighbors_sklearn,
        ),
        "sklearn.neighbors.KNeighborsRegressor": (
            neighbors_module,
            "KNeighborsRegressor",
            KNeighborsRegressor_daal4py,
            KNeighborsRegressor_sklearn,
        ),
        "sklearn.ensemble.RandomForestClassifier": (
            ensemble_module,
            "RandomForestClassifier",
            RandomForestClassifier_daal4py,
            RandomForestClassifier_sklearn,
        ),
        "sklearn.ensemble.RandomForestRegressor": (
            ensemble_module,
            "RandomForestRegressor",
            RandomForestRegressor_daal4py,
            RandomForestRegressor_sklearn,
        ),
        "sklearn.model_selection.train_test_split": (
            model_selection,
            "train_test_split",
            train_test_split,
            train_test_split_sklearn,
        ),
        "sklearn.utils.validation._assert_all_finite": (
            validation,
            "_assert_all_finite",
            _assert_all_finite,
            _assert_all_finite_sklearn,
        ),
        "sklearn.metrics.roc_auc_score": (
            metrics,
            "roc_auc_score",
            roc_auc_score,
            roc_auc_score_sklearn,
        ),
        "sklearn.manifold.TSNE": (manifold_module, "TSNE", TSNE_daal4py, TSNE_sklearn),
    }
    return mapping


# Algorithm map can be generated through '_get_map_of_algorithms'
def do_patch(name: str, map: PatchMap) -> None:
    descriptor = map.get(name)
    if descriptor is None:
        raise ValueError("Has no patch for: " + name)
    which, what, replacer, _ = descriptor
    setattr(which, what, replacer)


def do_unpatch(name: str, map: PatchMap) -> None:
    descriptor = map.get(name)
    if descriptor is None:
        raise ValueError("Has no patch for: " + name)
    which, what, _, replacer = descriptor
    if replacer is not None:
        setattr(which, what, replacer)
    elif hasattr(which, what):
        delattr(which, what)


def enable(
    name: Optional[str] = None,
    verbose: bool = True,
    deprecation: bool = True,
    map: Optional[PatchMap] = None,
):
    if map is None:
        map = _get_map_of_algorithms()
    if name is not None:
        do_patch(name, map=map)
    else:
        for key in map:
            do_patch(key, map=map)
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


def disable(name: Optional[str] = None, map: Optional[PatchMap] = None):
    if map is None:
        map = _get_map_of_algorithms()
    if name is not None:
        do_unpatch(name, map=map)
    else:
        for key in map:
            do_unpatch(key, map=map)


def _is_enabled(name: str, map: PatchMap) -> bool:
    descriptor = map.get(name)
    if descriptor is None:
        return False
    which, what, _, replacer = descriptor
    current = getattr(which, what, None)
    if current is None:
        return False
    return current == replacer


def patch_is_enabled(
    name: Optional[str] = None, return_map: bool = False, *, map: PatchMap
) -> Union[bool, dict[str, bool]]:
    if name is not None:
        return _is_enabled(name, map=map)
    if return_map:
        enabled = {}
        for key in map:
            enabled[key] = _is_enabled(key, map=map)
    else:
        enabled = True
        for key in map:
            enabled = enabled and _is_enabled(key, map=map)
    return enabled


def _patch_names() -> list[str]:
    return list(_get_map_of_algorithms().keys())
