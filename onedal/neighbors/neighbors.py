# ==============================================================================
# Copyright 2022 Intel Corporation
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

from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
import sys

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from .._config import _get_config
from ..common._estimator_checks import _check_is_fitted, _is_classifier, _is_regressor
from ..common._mixin import ClassifierMixin, RegressorMixin
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import (
    _check_array,
    _check_classification_targets,
    _check_n_features,
    _check_X_y,
    _column_or_1d,
    _num_samples,
)


class NeighborsCommonBase(metaclass=ABCMeta):
    def __init__(self):
        self.requires_y = False
        self.n_neighbors = None
        self.metric = None
        self.classes_ = None
        self.effective_metric_ = None
        self._fit_method = None
        self.radius = None
        self.effective_metric_params_ = None
        self._onedal_model = None

    def _parse_auto_method(self, method, n_samples, n_features):
        result_method = method

        if method in ["auto", "ball_tree"]:
            condition = (
                self.n_neighbors is not None and self.n_neighbors >= n_samples // 2
            )
            if self.metric == "precomputed" or n_features > 15 or condition:
                result_method = "brute"
            else:
                if self.metric == "euclidean":
                    result_method = "kd_tree"
                else:
                    result_method = "brute"

        return result_method

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def infer(self, *args, **kwargs): ...

    @abstractmethod
    def _onedal_fit(self, X, y): ...

    def _validate_data(
        self, X, y=None, reset=True, validate_separately=None, **check_params
    ):
        if y is None:
            if self.requires_y:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            X = _check_array(X, **check_params)
            out = X, y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling _check_array()
                # on X and y isn't equivalent to just calling _check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = _check_array(X, **check_X_params)
                y = _check_array(y, **check_y_params)
            else:
                X, y = _check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get("ensure_2d", True):
            _check_n_features(self, X, reset=reset)

        return out

    def _get_weights(self, dist, weights):
        if weights in (None, "uniform"):
            return None
        if weights == "distance":
            # if user attempts to classify a point that was zero distance from one
            # or more training points, those training points are weighted as 1.0
            # and the other points as 0.0
            if dist.dtype is np.dtype(object):
                for point_dist_i, point_dist in enumerate(dist):
                    # check if point_dist is iterable
                    # (ex: RadiusNeighborClassifier.predict may set an element of
                    # dist to 1e-6 to represent an 'outlier')
                    if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                        dist[point_dist_i] = point_dist == 0.0
                    else:
                        dist[point_dist_i] = 1.0 / point_dist
            else:
                with np.errstate(divide="ignore"):
                    dist = 1.0 / dist
                inf_mask = np.isinf(dist)
                inf_row = np.any(inf_mask, axis=1)
                dist[inf_row] = inf_mask[inf_row]
            return dist
        elif callable(weights):
            return weights(dist)
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )

    def _get_onedal_params(self, X, y=None, n_neighbors=None):
        class_count = 0 if self.classes_ is None else len(self.classes_)
        weights = getattr(self, "weights", "uniform")
        if self.effective_metric_ == "manhattan":
            p = 1.0
        elif self.effective_metric_ == "euclidean":
            p = 2.0
        else:
            p = self.p
        return {
            "fptype": X.dtype,
            "vote_weights": "uniform" if weights == "uniform" else "distance",
            "method": self._fit_method,
            "radius": self.radius,
            "class_count": class_count,
            "neighbor_count": self.n_neighbors if n_neighbors is None else n_neighbors,
            "metric": self.effective_metric_,
            "p": p,
            "metric_params": self.effective_metric_params_,
            "result_option": "indices|distances" if y is None else "responses",
        }


class NeighborsBase(NeighborsCommonBase, metaclass=ABCMeta):
    def __init__(
        self,
        n_neighbors=None,
        radius=None,
        algorithm="auto",
        metric="minkowski",
        p=2,
        metric_params=None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _validate_targets(self, y, dtype):
        arr = _column_or_1d(y, warn=True)

        try:
            return arr.astype(dtype, copy=False)
        except ValueError:
            return arr

    # REFACTOR NOTE: _validate_n_classes moved to sklearnex/neighbors/common.py
    # This method is no longer used in the onedal layer - all validation happens in sklearnex
    # Commented out for reference only
    # def _validate_n_classes(self):
    #     length = 0 if self.classes_ is None else len(self.classes_)
    #     if length < 2:
    #         raise ValueError(
    #             f"The number of classes has to be greater than one; got {length}"
    #         )

    def _fit(self, X, y):
        print(f"DEBUG oneDAL _fit START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}, y type={type(y)}", file=sys.stderr)
        self._onedal_model = None
        self._tree = None
        self._shape = None
        # REFACTOR STEP 1: Don't reset classes_ - it may have been set by sklearnex layer
        # self.classes_ = None
        if not hasattr(self, 'classes_'):
            self.classes_ = None
        self.effective_metric_ = getattr(self, "effective_metric_", self.metric)
        self.effective_metric_params_ = getattr(
            self, "effective_metric_params_", self.metric_params
        )

        _, xp, _ = _get_sycl_namespace(X)
        # REFACTOR: _validate_data call commented out - validation now happens in sklearnex layer
        # Original code kept for reference:
        # use_raw_input = _get_config().get("use_raw_input", False) is True
        if y is not None or self.requires_y:
            shape = getattr(y, "shape", None)
            # REFACTOR: _validate_data call commented out - validation now happens in sklearnex layer
            # if not use_raw_input:
            #     X, y = super()._validate_data(
            #         X, y, dtype=[np.float64, np.float32], accept_sparse="csr"
            #     )
            self._shape = shape if shape is not None else y.shape

            # REFACTOR: Classification target processing moved to sklearnex layer
            # This code is now commented out - processing MUST happen in sklearnex before calling fit
            # Assertion: Verify that sklearnex has done the preprocessing
            if _is_classifier(self):
                if not hasattr(self, 'classes_') or self.classes_ is None:
                    raise ValueError(
                        "Classification target processing must be done in sklearnex layer before calling onedal fit. "
                        "classes_ attribute is not set. This indicates the refactoring is incomplete."
                    )
                if not hasattr(self, '_y') or self._y is None:
                    raise ValueError(
                        "Classification target processing must be done in sklearnex layer before calling onedal fit. "
                        "_y attribute is not set. This indicates the refactoring is incomplete."
                    )
                print(f"DEBUG oneDAL: Using pre-processed classification targets from sklearnex (classes_={self.classes_})", file=sys.stderr)
            
            # Original classification processing code - NOW COMMENTED OUT (moved to sklearnex)
            # if _is_classifier(self):
            #     if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            #         self.outputs_2d_ = False
            #         y = y.reshape((-1, 1))
            #     else:
            #         self.outputs_2d_ = True

            #     _check_classification_targets(y)
            #     self.classes_ = []
            #     self._y = np.empty(y.shape, dtype=int)
            #     for k in range(self._y.shape[1]):
            #         classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            #         self.classes_.append(classes)

            #     if not self.outputs_2d_:
            #         self.classes_ = self.classes_[0]
            #         self._y = self._y.ravel()

            #     self._validate_n_classes()
            # else:
            else:
                # For regressors, just store y
                self._y = y
        # REFACTOR: _validate_data call commented out - validation now happens in sklearnex layer
        # elif not use_raw_input:
        #     X, _ = super()._validate_data(X, dtype=[np.float64, np.float32])

        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self._fit_X = X

        # REFACTOR: n_neighbors validation commented out - should be done in sklearnex layer
        # Original code kept for reference:
        # if self.n_neighbors is not None:
        #     if self.n_neighbors <= 0:
        #         raise ValueError("Expected n_neighbors > 0. Got %d" % self.n_neighbors)
        #     if not isinstance(self.n_neighbors, Integral):
        #         raise TypeError(
        #             "n_neighbors does not take %s value, "
        #             "enter integer value" % type(self.n_neighbors)
        #         )

        self._fit_method = super()._parse_auto_method(
            self.algorithm, self.n_samples_fit_, self.n_features_in_
        )

        _fit_y = None
        queue = QM.get_global_queue()
        gpu_device = queue is not None and queue.sycl_device.is_gpu

        print(f"DEBUG oneDAL _fit: Before _onedal_fit, X type={type(X)}, _fit_y type={type(_fit_y)}", file=sys.stderr)
        if _is_classifier(self) or (_is_regressor(self) and gpu_device):
            _fit_y = self._validate_targets(self._y, X.dtype).reshape((-1, 1))
        result = self._onedal_fit(X, _fit_y)
        print(f"DEBUG oneDAL _fit: After _onedal_fit, self._fit_X type={type(self._fit_X)}, shape={getattr(self._fit_X, 'shape', 'NO_SHAPE')}", file=sys.stderr)

        if y is not None and _is_regressor(self):
            self._y = y if self._shape is None else xp.reshape(y, self._shape)

        self._onedal_model = result
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # REFACTOR: Feature count validation commented out - should be done in sklearnex layer
        # Original validation code kept for reference:
        # use_raw_input = _get_config().get("use_raw_input", False) is True
        # n_features = getattr(self, "n_features_in_", None)
        # shape = getattr(X, "shape", None)
        # if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        #     raise ValueError(
        #         (
        #             f"X has {X.shape[1]} features, "
        #             f"but kneighbors is expecting "
        #             f"{n_features} features as input"
        #         )
        #     )
        
        # Still need n_features for _parse_auto_method call later
        n_features = getattr(self, "n_features_in_", None)

        _check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        # REFACTOR: n_neighbors validation commented out - should be done in sklearnex layer
        # Original validation code kept for reference:
        # elif n_neighbors <= 0:
        #     raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        # else:
        #     if not isinstance(n_neighbors, Integral):
        #         raise TypeError(
        #             "n_neighbors does not take %s value, "
        #             "enter integer value" % type(n_neighbors)
        #         )

        # REFACTOR: X array validation commented out - should be done in sklearnex layer
        # Original validation code kept for reference:
        # if X is not None:
        #     query_is_train = False
        #     if not use_raw_input:
        #         X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        # else:
        #     query_is_train = True
        #     X = self._fit_X
        #     # Include an extra neighbor to account for the sample itself being
        #     # returned, which is removed later
        #     n_neighbors += 1
        
        if X is not None:
            query_is_train = False
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            if query_is_train:
                n_neighbors -= 1  # ok to modify inplace because an error is raised
                inequality_str = "n_neighbors < n_samples_fit"
            else:
                inequality_str = "n_neighbors <= n_samples_fit"
            raise ValueError(
                f"Expected {inequality_str}, but "
                f"n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, "
                f"n_samples = {X.shape[0]}"  # include n_samples for common tests
            )

        chunked_results = None
        method = self._parse_auto_method(
            self._fit_method, self.n_samples_fit_, n_features
        )

        params = super()._get_onedal_params(X, n_neighbors=n_neighbors)
        prediction_results = self._onedal_predict(self._onedal_model, X, params)
        distances = from_table(prediction_results.distances)
        indices = from_table(prediction_results.indices)

        if method == "kd_tree":
            for i in range(distances.shape[0]):
                seq = distances[i].argsort()
                indices[i] = indices[i][seq]
                distances[i] = distances[i][seq]

        if return_distance:
            results = distances, indices
        else:
            results = indices

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


class KNeighborsClassifier(NeighborsBase, ClassifierMixin):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        p=2,
        metric="minkowski",
        metric_params=None,
        **kwargs,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            **kwargs,
        )
        self.weights = weights

    # direct access to the backend model constructor
    @bind_default_backend("neighbors.classification")
    def model(self): ...

    # direct access to the backend model constructor
    @bind_default_backend("neighbors.classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.classification")
    def infer(self, *args, **kwargs): ...

    def _onedal_fit(self, X, y):
        # global queue is set as per user configuration (`target_offload`) or from data prior to calling this internal function
        queue = QM.get_global_queue()
        params = self._get_onedal_params(X, y)
        X_table, y_table = to_table(X, y, queue=queue)
        return self.train(params, X_table, y_table).model

    def _onedal_predict(self, model, X, params):
        X = to_table(X, queue=QM.get_global_queue())
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"
        params["fptype"] = X.dtype
        result = self.infer(params, model, X)

        return result

    @supports_queue
    def fit(self, X, y, queue=None):
        return self._fit(X, y)

    @supports_queue
    def predict(self, X, queue=None):
        print(f"DEBUG KNeighborsClassifier.predict START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        use_raw_input = _get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        onedal_model = getattr(self, "_onedal_model", None)
        n_features = getattr(self, "n_features_in_", None)
        n_samples_fit_ = getattr(self, "n_samples_fit_", None)
        shape = getattr(X, "shape", None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError(
                (
                    f"X has {X.shape[1]} features, "
                    f"but KNNClassifier is expecting "
                    f"{n_features} features as input"
                )
            )

        _check_is_fitted(self)

        self._fit_method = self._parse_auto_method(
            self.algorithm, n_samples_fit_, n_features
        )

        # REFACTOR NOTE: _validate_n_classes() is now called during fit in sklearnex layer
        # No need to validate again during predict
        # self._validate_n_classes()

        params = self._get_onedal_params(X)
        prediction_result = self._onedal_predict(onedal_model, X, params)
        responses = from_table(prediction_result.responses)

        result = self.classes_.take(np.asarray(responses.ravel(), dtype=np.intp))
        print(f"DEBUG KNeighborsClassifier.predict END: result type={type(result)}", file=sys.stderr)
        return result

    @supports_queue
    def predict_proba(self, X, queue=None):
        print(f"DEBUG KNeighborsClassifier.predict_proba START: X type={type(X)}", file=sys.stderr)
        neigh_dist, neigh_ind = self.kneighbors(X, queue=queue)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X if X is not None else self._fit_X)

        print(f"DEBUG predict_proba: Calling _get_weights", file=sys.stderr)
        weights = self._get_weights(neigh_dist, self.weights)
        if weights is None:
            print(f"DEBUG predict_proba: weights is None, using ones_like", file=sys.stderr)
            weights = np.ones_like(neigh_ind)
        else:
            print(f"DEBUG predict_proba: weights calculated, type={type(weights)}", file=sys.stderr)

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    @supports_queue
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return self._kneighbors(X, n_neighbors, return_distance)


class KNeighborsRegressor(NeighborsBase, RegressorMixin):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        p=2,
        metric="minkowski",
        metric_params=None,
        **kwargs,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            **kwargs,
        )
        self.weights = weights

    @bind_default_backend("neighbors.search", lookup_name="train")
    def train_search(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.search", lookup_name="infer")
    def infer_search(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.regression")
    def infer(self, *args, **kwargs): ...

    def _onedal_fit(self, X, y):
        # global queue is set as per user configuration (`target_offload`) or from data prior to calling this internal function
        queue = QM.get_global_queue()
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X_table, y)

        if gpu_device:
            return self.train(params, X_table, y_table).model
        else:
            return self.train_search(params, X_table).model

    def _onedal_predict(self, model, X, params):
        assert self._onedal_model is not None, "Model is not trained"

        # global queue is set as per user configuration (`target_offload`) or from data prior to calling this internal function
        queue = QM.get_global_queue()
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        X = to_table(X, queue=queue)

        if "responses" not in params["result_option"] and gpu_device:
            params["result_option"] += "|responses"
        params["fptype"] = X.dtype

        if gpu_device:
            return self.infer(params, self._onedal_model, X)
        else:
            return self.infer_search(params, self._onedal_model, X)

    @supports_queue
    def fit(self, X, y, queue=None):
        return self._fit(X, y)

    @supports_queue
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return self._kneighbors(X, n_neighbors, return_distance)

    def _predict_gpu(self, X):
        use_raw_input = _get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        onedal_model = getattr(self, "_onedal_model", None)
        n_features = getattr(self, "n_features_in_", None)
        n_samples_fit_ = getattr(self, "n_samples_fit_", None)
        shape = getattr(X, "shape", None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError(
                (
                    f"X has {X.shape[1]} features, "
                    f"but KNNClassifier is expecting "
                    f"{n_features} features as input"
                )
            )

        _check_is_fitted(self)

        self._fit_method = self._parse_auto_method(
            self.algorithm, n_samples_fit_, n_features
        )

        params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params)
        responses = from_table(prediction_result.responses)
        result = responses.ravel()

        return result

    def _predict_skl(self, X):
        print(f"DEBUG KNeighborsRegressor._predict_skl START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        neigh_dist, neigh_ind = self.kneighbors(X)

        print(f"DEBUG _predict_skl: Calling _get_weights", file=sys.stderr)
        weights = self._get_weights(neigh_dist, self.weights)
        print(f"DEBUG _predict_skl: weights result={type(weights) if weights is not None else 'None'}", file=sys.stderr)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        print(f"DEBUG KNeighborsRegressor._predict_skl END: y_pred type={type(y_pred)}", file=sys.stderr)
        return y_pred

    @supports_queue
    def predict(self, X, queue=None):
        print(f"DEBUG KNeighborsRegressor.predict START: X type={type(X)}, queue={queue}", file=sys.stderr)
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        is_uniform_weights = getattr(self, "weights", "uniform") == "uniform"
        print(f"DEBUG KNeighborsRegressor.predict: gpu_device={gpu_device}, is_uniform_weights={is_uniform_weights}", file=sys.stderr)
        if gpu_device and is_uniform_weights:
            result = self._predict_gpu(X)
        else:
            result = self._predict_skl(X)
        print(f"DEBUG KNeighborsRegressor.predict END: result type={type(result)}", file=sys.stderr)
        return result


class NearestNeighbors(NeighborsBase):
    def __init__(
        self,
        n_neighbors=5,
        *,
        algorithm="auto",
        p=2,
        metric="minkowski",
        metric_params=None,
        **kwargs,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            **kwargs,
        )
        self.requires_y = False

    @bind_default_backend("neighbors.search")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.search")
    def infer(self, *arg, **kwargs): ...

    def _onedal_fit(self, X, y):
        # global queue is set as per user configuration (`target_offload`) or from data prior to calling this internal function
        queue = QM.get_global_queue()
        params = self._get_onedal_params(X, y)
        X, y = to_table(X, y, queue=queue)
        return self.train(params, X).model

    def _onedal_predict(self, model, X, params):
        X = to_table(X, queue=QM.get_global_queue())

        params["fptype"] = X.dtype
        return self.infer(params, model, X)

    @supports_queue
    def fit(self, X, y=None, queue=None):
        return self._fit(X, y)

    @supports_queue
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return self._kneighbors(X, n_neighbors, return_distance)