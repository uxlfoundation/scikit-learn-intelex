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
from ..utils.validation import _num_samples


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

        # Handle different input types for dtype
        try:
            fptype = X.dtype
        except AttributeError:
            # For pandas DataFrames or other types without dtype attribute
            import numpy as np

            fptype = np.float64

        # _fit_method should be set by sklearnex level before calling oneDAL
        if not hasattr(self, "_fit_method") or self._fit_method is None:
            raise ValueError(
                "_fit_method must be set by sklearnex level before calling oneDAL. "
                "This indicates improper usage - oneDAL neighbors should not be called directly."
            )

        return {
            "fptype": fptype,
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
        print(f"DEBUG oneDAL _fit START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  y type: {type(y)}, y shape: {getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        self._onedal_model = None
        self._tree = None
        self._shape = None
        self.classes_ = None
        self.effective_metric_ = getattr(self, "effective_metric_", self.metric)
        self.effective_metric_params_ = getattr(
            self, "effective_metric_params_", self.metric_params
        )

        _, xp, _ = _get_sycl_namespace(X)
        if y is not None or self.requires_y:
            shape = getattr(y, "shape", None)
            self._shape = shape if shape is not None else y.shape

            if _is_classifier(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                self.classes_ = []
                self._y = np.empty(y.shape, dtype=int)
                for k in range(self._y.shape[1]):
                    classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
                    self.classes_.append(classes)

                if not self.outputs_2d_:
                    self.classes_ = self.classes_[0]
                    self._y = self._y.ravel()
            else:
                self._y = y

        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        
        print(f"DEBUG oneDAL _fit BEFORE setting _fit_X:", file=sys.stderr)
        print(f"  X type: {type(X)}, isinstance(X, tuple): {isinstance(X, tuple)}", file=sys.stderr)
        
        # CRITICAL FIX: Ensure _fit_X is always an array, never a tuple
        # This is essential because sklearn's _fit method reads from self._fit_X directly
        if isinstance(X, tuple):
            print(f"DEBUG oneDAL _fit: X is tuple, extracting first element: {type(X)}", file=sys.stderr)
            # Extract the actual array from tuple created by from_table/to_table
            self._fit_X = X[0] if X[0] is not None else X[1]
        else:
            self._fit_X = X
            
        print(f"DEBUG oneDAL _fit AFTER setting _fit_X:", file=sys.stderr)
        print(f"  self._fit_X type: {type(self._fit_X)}, shape: {getattr(self._fit_X, 'shape', 'NO_SHAPE')}", file=sys.stderr)

        _fit_y = None
        queue = QM.get_global_queue()
        gpu_device = queue is not None and queue.sycl_device.is_gpu

        print(f"DEBUG oneDAL _fit BEFORE calling _onedal_fit:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  _fit_y type: {type(_fit_y)}, _fit_y shape: {getattr(_fit_y, 'shape', 'NO_SHAPE')}", file=sys.stderr)

        if _is_classifier(self) or (_is_regressor(self) and gpu_device):
            _fit_y = y.astype(X.dtype).reshape((-1, 1)) if y is not None else None
        result = self._onedal_fit(X, _fit_y)
        
        print(f"DEBUG oneDAL _fit AFTER _onedal_fit:", file=sys.stderr)
        print(f"  self._fit_X type: {type(self._fit_X)}, shape: {getattr(self._fit_X, 'shape', 'NO_SHAPE')}", file=sys.stderr)

        if y is not None and _is_regressor(self):
            self._y = y if self._shape is None else xp.reshape(y, self._shape)

        self._onedal_model = result
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        _check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X is not None:
            query_is_train = False
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        n_samples_fit = self.n_samples_fit_

        chunked_results = None
        # Use the fit method determined at sklearnex level
        method = getattr(self, "_fit_method", "brute")

        params = self._get_onedal_params(X, n_neighbors=n_neighbors)
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
        onedal_model = getattr(self, "_onedal_model", None)
        _check_is_fitted(self)

        params = self._get_onedal_params(X)
        prediction_result = self._onedal_predict(onedal_model, X, params)
        responses = from_table(prediction_result.responses)

        result = self.classes_.take(np.asarray(responses.ravel(), dtype=np.intp))
        return result

    @supports_queue
    def predict_proba(self, X, queue=None):
        neigh_dist, neigh_ind = self.kneighbors(X, queue=queue)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X if X is not None else self._fit_X)

        # Use uniform weights for now - weights calculation should be done at sklearnex level
        weights = np.ones_like(neigh_ind)

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
        onedal_model = getattr(self, "_onedal_model", None)
        _check_is_fitted(self)

        params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params)
        responses = from_table(prediction_result.responses)
        result = responses.ravel()

        return result

    def _predict_skl(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)

        # Use uniform weights for now - weights calculation should be done at sklearnex level
        weights = None

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

        return y_pred

    @supports_queue
    def predict(self, X, queue=None):
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        is_uniform_weights = getattr(self, "weights", "uniform") == "uniform"
        if gpu_device and is_uniform_weights:
            return self._predict_gpu(X)
        else:
            return self._predict_skl(X)


class NearestNeighbors(NeighborsBase):
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

    @bind_default_backend("neighbors.search")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("neighbors.search")
    def infer(self, *arg, **kwargs): ...

    def _onedal_fit(self, X, y):
        print(f"DEBUG NearestNeighbors _onedal_fit START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  y type: {type(y)}, y shape: {getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  self._fit_X BEFORE to_table: type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        
        # global queue is set as per user configuration (`target_offload`) or from data prior to calling this internal function
        queue = QM.get_global_queue()
        params = self._get_onedal_params(X, y)
        
        print(f"DEBUG NearestNeighbors _onedal_fit BEFORE to_table:", file=sys.stderr)
        print(f"  X type: {type(X)}, isinstance(X, tuple): {isinstance(X, tuple)}", file=sys.stderr)
        print(f"  y type: {type(y)}, isinstance(y, tuple): {isinstance(y, tuple)}", file=sys.stderr)
        
        X, y = to_table(X, y, queue=queue)
        
        print(f"DEBUG NearestNeighbors _onedal_fit AFTER to_table - CRITICAL POINT:", file=sys.stderr)
        print(f"  X type: {type(X)}, isinstance(X, tuple): {isinstance(X, tuple)}", file=sys.stderr)
        print(f"  y type: {type(y)}, isinstance(y, tuple): {isinstance(y, tuple)}", file=sys.stderr)
        print(f"  self._fit_X AFTER to_table: type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        
        result = self.train(params, X).model
        
        print(f"DEBUG NearestNeighbors _onedal_fit AFTER train:", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        
        return result

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