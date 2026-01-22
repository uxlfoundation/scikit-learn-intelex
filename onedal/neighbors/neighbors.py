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

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from ..common._estimator_checks import _check_is_fitted, _is_classifier, _is_regressor
from ..common._mixin import ClassifierMixin, RegressorMixin
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace


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

    def _fit(self, X, y):
        self._onedal_model = None
        self._tree = None
        # REFACTOR: Shape processing moved to sklearnex layer
        # _shape should be set by _process_classification_targets or _process_regression_targets in sklearnex
        # self._shape = None
        if not hasattr(self, "_shape"):
            self._shape = None
        # REFACTOR STEP 1: Don't reset classes_ - it may have been set by sklearnex layer
        # self.classes_ = None
        if not hasattr(self, "classes_"):
            self.classes_ = None
        self.effective_metric_ = getattr(self, "effective_metric_", self.metric)
        self.effective_metric_params_ = getattr(
            self, "effective_metric_params_", self.metric_params
        )

        # REFACTOR: _validate_data call commented out - validation now happens in sklearnex layer
        # Original code kept for reference:
        # use_raw_input = _get_config().get("use_raw_input", False) is True
        if y is not None or self.requires_y:
            # REFACTOR: Classification target processing moved to sklearnex layer
            # This code is now commented out - processing MUST happen in sklearnex before calling fit
            # Assertion: Verify that sklearnex has done the preprocessing
            if _is_classifier(self):
                # if not hasattr(self, "classes_") or self.classes_ is None:
                #     raise ValueError(
                #         "Classification target processing must be done in sklearnex layer before calling onedal fit. "
                #         "classes_ attribute is not set. This indicates the refactoring is incomplete."
                #     )
                if not hasattr(self, "_y") or self._y is None:
                    raise ValueError(
                        "Classification target processing must be done in sklearnex layer before calling onedal fit. "
                        "_y attribute is not set. This indicates the refactoring is incomplete."
                    )
            else:
                # For regressors, just store y
                self._y = y
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self._fit_X = X
        self._fit_method = super()._parse_auto_method(
            self.algorithm, self.n_samples_fit_, self.n_features_in_
        )

        _fit_y = None
        queue = QM.get_global_queue()
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        # Backend-specific formatting: GPU backend needs y in (-1, 1) shape
        if _is_classifier(self) or (_is_regressor(self) and gpu_device):
            _, xp, _ = _get_sycl_namespace(X)
            _fit_y = xp.reshape(self._y, (-1, 1))
        result = self._onedal_fit(X, _fit_y)

        # Original logic: reassign self._y for regressors after fit
        if y is not None and _is_regressor(self):
            _, xp, _ = _get_sycl_namespace(X)
            self._y = y if self._shape is None else xp.reshape(y, self._shape)

        self._onedal_model = result
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        _check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        n_features = getattr(self, "n_features_in_", None)

        if X is not None:
            query_is_train = False
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
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
        # REFACTOR: Convert to table FIRST, then get params from table (following PCA pattern)
        # This ensures dtype is normalized (array API dtype -> numpy dtype)
        X_table, y_table = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X_table, y)
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

    # REFACTOR: Keep _predict_gpu for GPU backend support (called by sklearnex)
    # This is the ONLY prediction method needed in onedal - it calls the backend directly
    # All computation logic (weights, averaging, etc.) is in sklearnex
    def _predict_gpu(self, X):
        # REFACTOR: Validation commented out - should be done in sklearnex layer before calling this
        onedal_model = getattr(self, "_onedal_model", None)
        n_features = getattr(self, "n_features_in_", None)
        n_samples_fit_ = getattr(self, "n_samples_fit_", None)

        _check_is_fitted(self)

        self._fit_method = self._parse_auto_method(
            self.algorithm, n_samples_fit_, n_features
        )

        params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params)
        responses = from_table(prediction_result.responses)
        result = responses.ravel()

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
        # REFACTOR: Convert to table FIRST, then get params from table (following PCA pattern)
        # This ensures dtype is normalized (array API dtype -> numpy dtype)
        # Note: NearestNeighbors has no y, so only convert X to avoid y becoming a table
        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X, y)
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