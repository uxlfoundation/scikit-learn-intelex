# ==============================================================================
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
# ==============================================================================

# daal4py KNN scikit-learn-compatible base classes

import logging
import numbers
import warnings

import numpy as np
from scipy import sparse as sp
from sklearn.base import is_classifier, is_regressor
from sklearn.neighbors import VALID_METRICS
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighborsBase
from sklearn.neighbors._base import RadiusNeighborsMixin as BaseRadiusNeighborsMixin
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import daal4py as d4p

from .._utils import (
    PatchingConditionsChain,
    get_patch_message,
    getFPType,
    sklearn_check_version,
)
from ..utils.validation import check_feature_names, check_n_features, get_requires_y_tag

if not sklearn_check_version("1.2"):
    from sklearn.neighbors._base import _check_weights


def training_algorithm(method, fptype, params):
    if method == "brute":
        train_alg = d4p.bf_knn_classification_training

    else:
        train_alg = d4p.kdtree_knn_classification_training

    params["fptype"] = fptype
    return train_alg(**params)


def prediction_algorithm(method, fptype, params):
    if method == "brute":
        predict_alg = d4p.bf_knn_classification_prediction
    else:
        predict_alg = d4p.kdtree_knn_classification_prediction

    params["fptype"] = fptype
    return predict_alg(**params)


def parse_auto_method(estimator, method, n_samples, n_features):
    result_method = method

    if method in ["auto", "ball_tree"]:
        condition = (
            estimator.n_neighbors is not None
            and estimator.n_neighbors >= estimator.n_samples_fit_ // 2
        )
        if estimator.metric == "precomputed" or n_features > 11 or condition:
            result_method = "brute"
        else:
            if estimator.effective_metric_ in VALID_METRICS["kd_tree"]:
                result_method = "kd_tree"
            else:
                result_method = "brute"

    return result_method


def daal4py_fit(estimator, X, fptype):
    estimator._fit_X = X
    estimator._fit_method = estimator.algorithm
    estimator.effective_metric_ = "euclidean"
    estimator._tree = None
    weights = getattr(estimator, "weights", "uniform")

    params = {
        "method": "defaultDense",
        "k": estimator.n_neighbors,
        "voteWeights": "voteUniform" if weights == "uniform" else "voteDistance",
        "resultsToCompute": "computeIndicesOfNeighbors|computeDistances",
        "resultsToEvaluate": (
            "none" if getattr(estimator, "_y", None) is None else "computeClassLabels"
        ),
    }
    if hasattr(estimator, "classes_"):
        params["nClasses"] = len(estimator.classes_)

    if getattr(estimator, "_y", None) is None:
        labels = None
    else:
        labels = estimator._y.reshape(-1, 1)

    method = parse_auto_method(
        estimator, estimator.algorithm, estimator.n_samples_fit_, estimator.n_features_in_
    )
    estimator._fit_method = method
    train_alg = training_algorithm(method, fptype, params)
    estimator._daal_model = train_alg.compute(X, labels).model


def daal4py_kneighbors(estimator, X=None, n_neighbors=None, return_distance=True):
    n_features = getattr(estimator, "n_features_in_", None)
    shape = getattr(X, "shape", None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError(
            (
                f"X has {X.shape[1]} features, "
                f"but kneighbors is expecting {n_features} features as input"
            )
        )

    check_is_fitted(estimator)

    if n_neighbors is None:
        n_neighbors = estimator.n_neighbors
    elif n_neighbors <= 0:
        raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
    else:
        if not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, "
                "enter integer value" % type(n_neighbors)
            )

    if X is not None:
        query_is_train = False
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    else:
        query_is_train = True
        X = estimator._fit_X
        # Include an extra neighbor to account for the sample itself being
        # returned, which is removed later
        n_neighbors += 1

    n_samples_fit = estimator.n_samples_fit_
    if n_neighbors > n_samples_fit:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" % (n_samples_fit, n_neighbors)
        )

    chunked_results = None

    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    weights = getattr(estimator, "weights", "uniform")

    params = {
        "method": "defaultDense",
        "k": n_neighbors,
        "voteWeights": "voteUniform" if weights == "uniform" else "voteDistance",
        "resultsToCompute": "computeIndicesOfNeighbors|computeDistances",
        "resultsToEvaluate": (
            "none" if getattr(estimator, "_y", None) is None else "computeClassLabels"
        ),
    }
    if hasattr(estimator, "classes_"):
        params["nClasses"] = len(estimator.classes_)

    method = parse_auto_method(
        estimator, estimator._fit_method, estimator.n_samples_fit_, n_features
    )

    predict_alg = prediction_algorithm(method, fptype, params)
    prediction_result = predict_alg.compute(X, estimator._daal_model)

    distances = prediction_result.distances
    indices = prediction_result.indices

    if method == "kd_tree":
        for i in range(distances.shape[0]):
            seq = distances[i].argsort()
            indices[i] = indices[i][seq]
            distances[i] = distances[i][seq]

    if return_distance:
        results = distances, indices.astype(int)
    else:
        results = indices.astype(int)

    if chunked_results is not None:
        if return_distance:
            neigh_dist, neigh_ind = zip(*chunked_results)
            results = np.vstack(neigh_dist), np.vstack(neigh_ind)
        else:
            results = np.vstack(chunked_results)

    if not query_is_train:
        return results
    # If the query data is the same as the indexed data, we would like
    # to ignore the first nearest neighbor of every sample, i.e
    # the sample itself.
    if return_distance:
        neigh_dist, neigh_ind = results
    else:
        neigh_ind = results

    n_queries, _ = X.shape
    sample_range = np.arange(n_queries)[:, None]
    sample_mask = neigh_ind != sample_range

    # Corner case: When the number of duplicates are more
    # than the number of neighbors, the first NN will not
    # be the sample, but a duplicate.
    # In that case mask the first duplicate.
    dup_gr_nbrs = np.all(sample_mask, axis=1)
    sample_mask[:, 0][dup_gr_nbrs] = False
    neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))

    if return_distance:
        neigh_dist = np.reshape(neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
        return neigh_dist, neigh_ind
    return neigh_ind


def validate_data(
    estimator, X, y=None, reset=True, validate_separately=False, **check_params
):
    if y is None:
        requires_y = get_requires_y_tag(estimator)

        if requires_y:
            raise ValueError(
                f"This {estimator.__class__.__name__} estimator "
                f"requires y to be passed, but the target y is None."
            )
        X = check_array(X, **check_params)
        out = X, y
    else:
        if validate_separately:
            # We need this because some estimators validate X and y
            # separately, and in general, separately calling check_array()
            # on X and y isn't equivalent to just calling check_X_y()
            # :(
            check_X_params, check_y_params = validate_separately
            X = check_array(X, **check_X_params)
            y = check_array(y, **check_y_params)
        else:
            X, y = check_X_y(X, y, **check_params)
        out = X, y

    if check_params.get("ensure_2d", True):
        check_n_features(estimator, X, reset=reset)

    return out


class NeighborsBase(BaseNeighborsBase):
    def __init__(
        self,
        n_neighbors=None,
        radius=None,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y=None):
        if self.metric_params is not None and "p" in self.metric_params:
            if self.p is not None:
                warnings.warn(
                    "Parameter p is found in metric_params. "
                    "The corresponding parameter from __init__ "
                    "is ignored.",
                    SyntaxWarning,
                    stacklevel=2,
                )

        if hasattr(self, "weights") and not sklearn_check_version("1.2"):
            self.weights = _check_weights(self.weights)

        check_feature_names(self, X, reset=True)

        X_incorrect_type = isinstance(
            X, (KDTree, BallTree, NeighborsBase, BaseNeighborsBase)
        )
        single_output = True
        self._daal_model = None
        shape = None
        correct_n_classes = True

        requires_y = get_requires_y_tag(self)

        if y is not None or requires_y:
            if not X_incorrect_type or y is None:
                X, y = validate_data(
                    self,
                    X,
                    y,
                    accept_sparse="csr",
                    multi_output=True,
                    dtype=[np.float64, np.float32],
                )
                single_output = False if y.ndim > 1 and y.shape[1] > 1 else True

            shape = y.shape

            if is_classifier(self) or is_regressor(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                if is_classifier(self):
                    check_classification_targets(y)
                self.classes_ = []
                self._y = np.empty(y.shape, dtype=int)
                for k in range(self._y.shape[1]):
                    classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
                    self.classes_.append(classes)

                if not self.outputs_2d_:
                    self.classes_ = self.classes_[0]
                    self._y = self._y.ravel()

                n_classes = len(self.classes_)
                if n_classes < 2:
                    correct_n_classes = False
            else:
                self._y = y
        else:
            if not X_incorrect_type:
                X, _ = validate_data(
                    self, X, accept_sparse="csr", dtype=[np.float64, np.float32]
                )

        if not X_incorrect_type:
            self.n_samples_fit_ = X.shape[0]
            self.n_features_in_ = X.shape[1]

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        weights = getattr(self, "weights", "uniform")

        def stock_fit(self, X, y):
            result = super(NeighborsBase, self)._fit(X, y)
            return result

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError("Expected n_neighbors > 0. Got %d" % self.n_neighbors)
            if not isinstance(self.n_neighbors, numbers.Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(self.n_neighbors)
                )

        _patching_status = PatchingConditionsChain(
            "sklearn.neighbors.KNeighborsMixin.kneighbors"
        )
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    self.metric == "minkowski"
                    and self.p == 2
                    or self.metric == "euclidean",
                    f"'{self.metric}' (p={self.p}) metric is not supported. "
                    "Only 'euclidean' or 'minkowski' with p=2 metrics are supported.",
                ),
                (not X_incorrect_type, "X is not Tree or Neighbors instance or array."),
                (
                    weights in ["uniform", "distance"],
                    f"'{weights}' weights is not supported. "
                    "Only 'uniform' and 'distance' weights are supported.",
                ),
                (
                    self.algorithm in ["brute", "kd_tree", "auto", "ball_tree"],
                    f"'{self.algorithm}' algorithm is not supported. "
                    "Only 'brute', 'kd_tree', 'auto' and 'ball_tree' "
                    "algorithms are supported.",
                ),
                (single_output, "Multiple outputs are not supported."),
                (fptype is not None, "Unable to get dtype."),
                (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                (correct_n_classes, "Number of classes < 2."),
            ]
        )
        _patching_status.write_log()
        if _dal_ready:
            try:
                daal4py_fit(self, X, fptype)
                result = self
            except RuntimeError:
                logging.info(
                    "sklearn.neighbors.KNeighborsMixin."
                    "kneighbors: " + get_patch_message("sklearn_after_daal")
                )
                result = stock_fit(self, X, y)
        else:
            result = stock_fit(self, X, y)

        if y is not None and is_regressor(self):
            self._y = y if shape is None else y.reshape(shape)

        return result


class KNeighborsMixin(BaseKNeighborsMixin):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        daal_model = getattr(self, "_daal_model", None)
        if X is not None and self.metric != "precomputed":
            X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        x = self._fit_X if X is None else X
        try:
            fptype = getFPType(x)
        except ValueError:
            fptype = None

        _patching_status = PatchingConditionsChain(
            "sklearn.neighbors.KNeighborsMixin.kneighbors"
        )
        _dal_ready = _patching_status.and_conditions(
            [
                (daal_model is not None, "oneDAL model was not trained."),
                (fptype is not None, "Unable to get dtype."),
                (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            ]
        )
        _patching_status.write_log()

        if _dal_ready:
            result = daal4py_kneighbors(self, X, n_neighbors, return_distance)
        else:
            if (
                daal_model is not None
                or getattr(self, "_tree", 0) is None
                and self._fit_method == "kd_tree"
            ):
                BaseNeighborsBase._fit(self, self._fit_X, getattr(self, "_y", None))
            result = super(KNeighborsMixin, self).kneighbors(
                X, n_neighbors, return_distance
            )

        return result


class RadiusNeighborsMixin(BaseRadiusNeighborsMixin):
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        daal_model = getattr(self, "_daal_model", None)

        if (
            daal_model is not None
            or getattr(self, "_tree", 0) is None
            and self._fit_method == "kd_tree"
        ):
            BaseNeighborsBase._fit(self, self._fit_X, getattr(self, "_y", None))
        result = BaseRadiusNeighborsMixin.radius_neighbors(
            self, X, radius, return_distance, sort_results
        )

        return result
