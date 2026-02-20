# ==============================================================================
# Copyright 2021 Intel Corporation
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

from sklearn.metrics import r2_score
from sklearn.neighbors._regression import (
    KNeighborsRegressor as _sklearn_KNeighborsRegressor,
)
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.neighbors import KNeighborsRegressor as onedal_KNeighborsRegressor

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data
from .common import KNeighborsDispatchingBase, _convert_to_numpy


@enable_array_api("1.5")  # validate_data y_numeric requires sklearn >=1.5
@control_n_jobs(decorated_methods=["fit", "predict", "kneighbors", "score"])
class KNeighborsRegressor(KNeighborsDispatchingBase, _sklearn_KNeighborsRegressor):
    __doc__ = _sklearn_KNeighborsRegressor.__doc__
    # Default onedal estimator class - SPMD subclasses can override this
    _onedal_estimator = onedal_KNeighborsRegressor

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_KNeighborsRegressor._parameter_constraints
        }

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_KNeighborsRegressor.fit,
            },
            X,
            y,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)

        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_KNeighborsRegressor.predict,
            },
            X,
        )

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)

        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": _sklearn_KNeighborsRegressor.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)

        check_is_fitted(self)

        self._kneighbors_validation(X, n_neighbors)

        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": _sklearn_KNeighborsRegressor.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    def _onedal_fit(self, X, y, queue=None):
        xp, _ = get_namespace(X, y)

        if not get_config()["use_raw_input"]:
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                multi_output=True,
                y_numeric=True,
            )

        self._set_effective_metric()

        self._process_regression_targets(y)
        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        # Use class-level _onedal_estimator if available (for SPMD), else use module-level
        if hasattr(self.__class__, "_onedal_estimator"):
            self._onedal_estimator = self.__class__._onedal_estimator(**onedal_params)
        else:
            self._onedal_estimator = onedal_KNeighborsRegressor(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator._shape = self._shape
        self._onedal_estimator._y = self._y

        # GPU regression uses full train (needs y reshaped to (-1, 1))
        # CPU regression uses train_search (only needs X, y must be None)
        # Check queue first, then fall back to data's device (use_raw_input path)
        if queue is not None:
            gpu_device = getattr(queue.sycl_device, "is_gpu", False)
        elif hasattr(X, "sycl_queue"):
            gpu_device = getattr(X.sycl_queue.sycl_device, "is_gpu", False)
        else:
            gpu_device = False
        if gpu_device:
            fit_y = xp.reshape(y, (-1, 1))
        else:
            fit_y = None
        self._onedal_estimator.fit(X, fit_y, queue=queue)

        self._save_attributes()

    def _process_regression_targets(self, y):
        """Process regression targets and set shape-related attributes.

        Parameters
        ----------
        y : array-like
            Target values
        """
        shape = getattr(y, "shape", None)
        self._shape = shape if shape is not None else y.shape
        self._y = y

    def _onedal_predict(self, X, queue=None):
        # Dispatch between GPU and SKL prediction methods
        # Check queue first, then fall back to data's device (use_raw_input path)
        if queue is not None:
            gpu_device = getattr(queue.sycl_device, "is_gpu", False)
        elif hasattr(X, "sycl_queue"):
            gpu_device = getattr(X.sycl_queue.sycl_device, "is_gpu", False)
        else:
            gpu_device = False
        is_uniform_weights = getattr(self, "weights", "uniform") == "uniform"

        if gpu_device and is_uniform_weights:
            return self._predict_gpu(X, queue=queue)
        else:
            return self._predict_skl(X, queue=queue)

    def _predict_gpu(self, X, queue=None):
        """GPU prediction path - calls onedal backend."""
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )
        return self._onedal_estimator._predict_gpu(X)

    def _predict_skl_regression(self, X):
        """SKL prediction path for regression - calls kneighbors, computes predictions.

        This method handles X=None (LOOCV) properly by calling self.kneighbors which
        has the query_is_train logic.

        Parameters
        ----------
        X : array-like or None
            Query samples (or None for LOOCV).

        Returns
        -------
        array-like
            Predicted regression values.
        """
        neigh_dist, neigh_ind = self._onedal_estimator.kneighbors(X)
        return self._compute_weighted_prediction(
            neigh_dist, neigh_ind, self.weights, self._y
        )

    def _predict_skl(self, X, queue=None):
        """SKL prediction path - calls kneighbors through sklearnex, computes prediction here."""
        if X is not None and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self, X, dtype=[xp.float64, xp.float32], accept_sparse="csr", reset=False
            )
        return self._predict_skl_regression(X)

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Determine if query is the training data
        if X is not None:
            query_is_train = False
            if not get_config()["use_raw_input"]:
                xp, _ = get_namespace(X)
                X = validate_data(
                    self,
                    X,
                    dtype=[xp.float64, xp.float32],
                    accept_sparse="csr",
                    reset=False,
                )
        else:
            query_is_train = True
            X = self._fit_X

        # Resolve effective n_neighbors (adjust for self-exclusion)
        effective_n_neighbors = (
            n_neighbors if n_neighbors is not None else self.n_neighbors
        )
        if query_is_train:
            effective_n_neighbors += 1

        # Validate bounds with adjusted n_neighbors
        self._validate_kneighbors_bounds(effective_n_neighbors, query_is_train, X)

        # Always get both distances and indices for post-processing
        distances, indices = self._onedal_estimator.kneighbors(
            X, effective_n_neighbors, return_distance=True, queue=queue
        )

        return self._kneighbors_postprocess(
            distances,
            indices,
            n_neighbors if n_neighbors is not None else self.n_neighbors,
            return_distance,
            query_is_train,
        )

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            _convert_to_numpy(y),
            self._onedal_predict(X, queue=queue),
            sample_weight=_convert_to_numpy(sample_weight),
        )

    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_KNeighborsRegressor.__doc__
    predict.__doc__ = _sklearn_KNeighborsRegressor.predict.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsRegressor.kneighbors.__doc__
    score.__doc__ = _sklearn_KNeighborsRegressor.score.__doc__
