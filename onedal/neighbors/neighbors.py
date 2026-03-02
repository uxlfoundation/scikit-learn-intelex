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

from ..common._estimator_checks import _check_is_fitted, _is_classifier
from ..common._mixin import ClassifierMixin, RegressorMixin
from ..datatypes import from_table, to_table


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
        class_count = 0 if self.classes_ is None else self.classes_.shape[0]
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
        if not hasattr(self, "_shape"):
            self._shape = None
        if not hasattr(self, "classes_"):
            self.classes_ = None
        self.effective_metric_ = getattr(self, "effective_metric_", self.metric)
        self.effective_metric_params_ = getattr(
            self, "effective_metric_params_", self.metric_params
        )

        if y is not None or self.requires_y:
            if _is_classifier(self):
                if not hasattr(self, "_y") or self._y is None:
                    raise ValueError(
                        "Internal error: Classification target processing must be done in sklearnex layer before calling onedal fit. "
                        "_y attribute is not set."
                    )
            elif y is not None:
                # For regressors, store y only if provided
                self._y = y
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self._fit_X = X
        self._fit_method = super()._parse_auto_method(
            self.algorithm, self.n_samples_fit_, self.n_features_in_
        )

        result = self._onedal_fit(X, y)

        self._onedal_model = result
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Raw kneighbors: calls C++ backend and returns from_table results.

        All post-processing (kd_tree sorting, query_is_train self-exclusion)
        is handled in the sklearnex layer (_kneighbors_postprocess).

        Returns numpy arrays (standard intermediate format). The sklearnex
        layer's wrap_output_data converts to the user's expected type.
        """
        _check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X is None:
            X = self._fit_X

        params = super()._get_onedal_params(X, n_neighbors=n_neighbors)
        prediction_results = self._onedal_predict(self._onedal_model, X, params)
        distances = from_table(prediction_results.distances)
        indices = from_table(prediction_results.indices)

        if return_distance:
            return distances, indices
        return indices


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
        queue = QM.get_global_queue()
        X_table, _ = to_table(X, y, queue=queue)
        params = self._get_onedal_params(X_table, y)
        return self.train(params, X_table).model

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
