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

from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase
from onedal._device_offload import _transfer_to_host

@enable_array_api
@control_n_jobs(decorated_methods=["fit", "predict", "kneighbors", "score"])
class KNeighborsRegressor(KNeighborsDispatchingBase, _sklearn_KNeighborsRegressor):
    __doc__ = _sklearn_KNeighborsRegressor.__doc__
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

        result = dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_KNeighborsRegressor.predict,
            },
            X,
        )
        return result

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)

        result = dispatch(
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
        return result

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # Validate n_neighbors parameter first (before check_is_fitted)
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)

        check_is_fitted(self)

        # Validate kneighbors parameters (inherited from KNeighborsDispatchingBase)
        self._kneighbors_validation(X, n_neighbors)

        result = dispatch(
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
        return result

    def _onedal_fit(self, X, y, queue=None):
        xp, _ = get_namespace(X)
        # REFACTOR: Use validate_data to convert pandas to numpy and validate types for X only
        X = validate_data(
            self,
            X,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
        )
        # REFACTOR: Process regression targets in sklearnex before passing to onedal
        # This sets _shape and _y attributes
        self._process_regression_targets(y)
        
        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        self._onedal_estimator = onedal_KNeighborsRegressor(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_

        # REFACTOR: Pass pre-processed shape and _y to onedal
        # For GPU backend, reshape _y to (-1, 1) before passing to onedal
        from onedal.utils import _sycl_queue_manager as QM

        queue_instance = QM.get_global_queue()
        gpu_device = queue_instance is not None and queue_instance.sycl_device.is_gpu

        self._onedal_estimator._shape = self._shape
        # REFACTOR: Reshape _y for GPU backend (needs column vector)
        # Following PCA pattern: all data preparation in sklearnex
        if gpu_device:
            self._onedal_estimator._y = xp.reshape(self._y, (-1, 1))
        else:
            self._onedal_estimator._y = self._y
      
        self._onedal_estimator.fit(X, y, queue=queue)
        self._save_attributes()

        # REFACTOR: Replicate the EXACT post-fit reshaping from original onedal code
        # Original onedal code (after fit):
        #     if y is not None and _is_regressor(self):
        #         _, xp, _ = _get_sycl_namespace(X)
        #         self._y = y if self._shape is None else xp.reshape(y, self._shape)
        # Now doing this in sklearnex layer
        if y is not None:
            xp, _ = get_namespace(y)
            self._y = y if self._shape is None else xp.reshape(y, self._shape)
            # Also update the onedal estimator's _y since that's what gets used in predict
            self._onedal_estimator._y = self._y

    def _onedal_predict(self, X, queue=None):
        # Dispatch between GPU and SKL prediction methods
        # This logic matches onedal regressor predict() method but computation happens in sklearnex
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        is_uniform_weights = getattr(self, "weights", "uniform") == "uniform"

        if gpu_device and is_uniform_weights:
            # GPU path: call onedal backend directly
            result = self._predict_gpu(X, queue=queue)
        else:
            # SKL path: call kneighbors (through sklearnex) then compute in sklearnex
            result = self._predict_skl(X, queue=queue)
        return result

    def _predict_gpu(self, X, queue=None):
        """GPU prediction path - calls onedal backend."""
        # Call onedal backend for GPU prediction (X is already validated by predict())
        result = self._onedal_estimator._predict_gpu(X)
        return result

    def _predict_skl(self, X, queue=None):
        """SKL prediction path - calls kneighbors through sklearnex, computes prediction here."""
        # Use the unified helper from common.py (calls kneighbors + computes prediction)
        result = self._predict_skl_regression(X)
        return result

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Validate X to convert array API/pandas to numpy and check feature names (only if X is not None)
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        # REFACTOR: All post-processing now in sklearnex following PCA pattern
        # Prepare inputs and handle query_is_train case
        X, n_neighbors, query_is_train = self._prepare_kneighbors_inputs(X, n_neighbors)

        # Get raw results from onedal backend
        result = self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

        # Apply post-processing (kd_tree sorting, removing self from results)
        result = self._kneighbors_post_processing(
            X, n_neighbors, return_distance, result, query_is_train
        )
        return result

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        y_pred = self._onedal_predict(X, queue=queue)

        # Convert array API/USM arrays back to numpy for r2_score
        # r2_score doesn't support Array API, following PCA's pattern with _transfer_to_host
        _, host_data = _transfer_to_host(y, y_pred, sample_weight)
        y, y_pred, sample_weight = host_data

        result = r2_score(y, y_pred, sample_weight=sample_weight)
        return result

    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._y = self._onedal_estimator._y
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_KNeighborsRegressor.__doc__
    predict.__doc__ = _sklearn_KNeighborsRegressor.predict.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsRegressor.kneighbors.__doc__
    score.__doc__ = _sklearn_KNeighborsRegressor.score.__doc__
