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
from sklearn.utils.validation import assert_all_finite, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal._device_offload import _transfer_to_host
from onedal.neighbors import KNeighborsRegressor as onedal_KNeighborsRegressor
from onedal.utils import _sycl_queue_manager as QM

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase


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
        # Validate n_neighbors parameter first (before check_is_fitted)
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)

        check_is_fitted(self)

        # Validate kneighbors parameters (inherited from KNeighborsDispatchingBase)
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

        # Validation step - validates and converts dtypes to float32/float64
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

            # Set effective metric after validation
            self._set_effective_metric()
        else:
            # SPMD mode: skip validation but still set effective metric
            self._set_effective_metric()

        # Process regression targets before passing to onedal (uses validated y)
        self._process_regression_targets(y)

        # Call onedal backend
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
        self._onedal_estimator._shape = self._shape
        self._onedal_estimator._y = self._y

        # Pass validated X and y to onedal (after validate_data converted dtypes)
        # Note: onedal layer handles backend-specific reshape (GPU needs (-1,1) format)
        self._onedal_estimator.fit(X, y, queue=queue)

        # Post-processing: save attributes
        # Note: _y reshape now happens in onedal layer after fit (matches original main branch logic)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        # Dispatch between GPU and SKL prediction methods
        # This logic matches onedal regressor predict() method but computation happens in sklearnex
        # Note: X validation happens in kneighbors (for SKL path) or _predict_gpu (for GPU path)
        gpu_device = queue is not None and getattr(queue.sycl_device, "is_gpu", False)
        is_uniform_weights = getattr(self, "weights", "uniform") == "uniform"

        if gpu_device and is_uniform_weights:
            # GPU path: call onedal backend directly
            return self._predict_gpu(X, queue=queue)
        else:
            # SKL path: call kneighbors (through sklearnex) then compute in sklearnex
            return self._predict_skl(X, queue=queue)

    def _predict_gpu(self, X, queue=None):
        """GPU prediction path - calls onedal backend."""
        # Validate X for GPU path (SKL path validation happens in kneighbors)
        if X is not None:
            xp, _ = get_namespace(X)
            # For precomputed metric, only check NaN/inf, don't validate features
            if getattr(self, "effective_metric_", self.metric) == "precomputed":
                from ..utils.validation import assert_all_finite

                assert_all_finite(X, allow_nan=False, input_name="X")
            else:
                X = validate_data(
                    self,
                    X,
                    dtype=[xp.float64, xp.float32],
                    accept_sparse="csr",
                    reset=False,
                )
        # Call onedal backend for GPU prediction
        return self._onedal_estimator._predict_gpu(X)

    def _predict_skl(self, X, queue=None):
        """SKL prediction path - calls kneighbors through sklearnex, computes prediction here."""
        # Use the unified helper from common.py (calls kneighbors + computes prediction)
        return self._predict_skl_regression(X)

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Validation step
        if X is not None and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        # onedal backend now handles all logic:
        # - X=None case (query_is_train)
        # - kd_tree sorting
        # - removing self from results
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        y_pred = self._onedal_predict(X, queue=queue)

        # Convert array API/USM arrays back to numpy for r2_score
        # r2_score doesn't support Array API, following PCA's pattern with _transfer_to_host
        _, host_data = _transfer_to_host(y, y_pred, sample_weight)
        y, y_pred, sample_weight = host_data

        return r2_score(y, y_pred, sample_weight=sample_weight)

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
