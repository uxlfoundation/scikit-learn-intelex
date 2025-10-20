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

import numpy as np
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
        import sys
        print(f"DEBUG KNeighborsRegressor.fit START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}, y type={type(y)}", file=sys.stderr)
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
        print(f"DEBUG KNeighborsRegressor.fit END: _fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        return self

    @wrap_output_data
    def predict(self, X):
        import sys
        print(f"DEBUG KNeighborsRegressor.predict START: X type={type(X)}", file=sys.stderr)
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
        print(f"DEBUG KNeighborsRegressor.predict END: result type={type(result)}", file=sys.stderr)
        return result

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        import sys
        print(f"DEBUG KNeighborsRegressor.score START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
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
        print(f"DEBUG KNeighborsRegressor.score END: result={result}", file=sys.stderr)
        return result

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        import sys
        print(f"DEBUG KNeighborsRegressor.kneighbors START: X type={type(X)}, n_neighbors={n_neighbors}, return_distance={return_distance}", file=sys.stderr)
        
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
        print(f"DEBUG KNeighborsRegressor.kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_fit(self, X, y, queue=None):
        import sys
        print(f"DEBUG KNeighborsRegressor._onedal_fit START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
        
        # Get array namespace for array API support
        xp, _ = get_namespace(X)
        print(f"DEBUG: Array namespace: {xp}", file=sys.stderr)
        
        # REFACTOR: Use validate_data to convert pandas to numpy and validate types for X only
        # ensure_all_finite=False to allow nan_euclidean metric to work (will fallback to sklearn)
        X = validate_data(
            self, X, dtype=[xp.float64, xp.float32], accept_sparse="csr", ensure_all_finite=False
        )
        print(f"DEBUG: After validate_data, X type={type(X)}, y type={type(y)}", file=sys.stderr)
        
        # REFACTOR: Process regression targets in sklearnex before passing to onedal
        # This sets _shape and _y attributes
        print(f"DEBUG: Processing regression targets in sklearnex", file=sys.stderr)
        y_processed = self._process_regression_targets(y)
        print(f"DEBUG: After _process_regression_targets, _shape={self._shape}, _y type={type(self._y)}", file=sys.stderr)
        
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
        print(f"DEBUG: Set onedal_estimator._shape={self._onedal_estimator._shape}", file=sys.stderr)
        print(f"DEBUG: GPU device={gpu_device}, _y shape={self._onedal_estimator._y.shape}", file=sys.stderr)
        
        print(f"DEBUG KNeighborsRegressor._onedal_fit: Calling onedal_estimator.fit", file=sys.stderr)
        self._onedal_estimator.fit(X, y, queue=queue)
        print(f"DEBUG KNeighborsRegressor._onedal_fit: After fit, calling _save_attributes", file=sys.stderr)

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
            print(f"DEBUG: After reshape, self._y type={type(self._y)}, shape={getattr(self._y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        print(f"DEBUG KNeighborsRegressor._onedal_fit END: self._fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)

    def _onedal_predict(self, X, queue=None):
        import sys
        print(f"DEBUG KNeighborsRegressor._onedal_predict START: X type={type(X)}", file=sys.stderr)
        
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
        
        print(f"DEBUG KNeighborsRegressor._onedal_predict END: result type={type(result)}", file=sys.stderr)
        return result
    
    def _predict_gpu(self, X, queue=None):
        """GPU prediction path - calls onedal backend."""
        import sys
        print(f"DEBUG KNeighborsRegressor._predict_gpu START: X type={type(X)}", file=sys.stderr)
        # Call onedal backend for GPU prediction (X is already validated by predict())
        result = self._onedal_estimator._predict_gpu(X)
        print(f"DEBUG KNeighborsRegressor._predict_gpu END: result type={type(result)}", file=sys.stderr)
        return result
    
    def _predict_skl(self, X, queue=None):
        """SKL prediction path - calls kneighbors through sklearnex, computes prediction here."""
        import sys
        print(f"DEBUG KNeighborsRegressor._predict_skl START: X type={type(X)}", file=sys.stderr)
        
        # Use the unified helper from common.py (calls kneighbors + computes prediction)
        result = self._predict_skl_regression(X)
        
        print(f"DEBUG KNeighborsRegressor._predict_skl END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        import sys
        print(f"DEBUG KNeighborsRegressor._onedal_kneighbors START: X type={type(X)}, n_neighbors={n_neighbors}, return_distance={return_distance}", file=sys.stderr)
        
        # Validate X to convert array API/pandas to numpy and check feature names (only if X is not None)
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self, X, dtype=[xp.float64, xp.float32], accept_sparse="csr", reset=False, ensure_all_finite=False
            )
        
        # REFACTOR: All post-processing now in sklearnex following PCA pattern
        # Prepare inputs and handle query_is_train case
        X, n_neighbors, query_is_train = self._prepare_kneighbors_inputs(X, n_neighbors)
        
        # Get raw results from onedal backend
        result = self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )
        
        # Apply post-processing (kd_tree sorting, removing self from results)
        result = self._kneighbors_post_processing(X, n_neighbors, return_distance, result, query_is_train)
        
        print(f"DEBUG KNeighborsRegressor._onedal_kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        import sys
        print(f"DEBUG KNeighborsRegressor._onedal_score START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
        result = r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )
        print(f"DEBUG KNeighborsRegressor._onedal_score END: result={result}", file=sys.stderr)
        return result

    def _save_attributes(self):
        import sys
        print(f"DEBUG KNeighborsRegressor._save_attributes START", file=sys.stderr)
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        print(f"DEBUG KNeighborsRegressor._save_attributes: _fit_X type={type(self._fit_X)}", file=sys.stderr)
        self._y = self._onedal_estimator._y
        print(f"DEBUG KNeighborsRegressor._save_attributes: _y type={type(self._y)}", file=sys.stderr)
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree
        print(f"DEBUG KNeighborsRegressor._save_attributes END", file=sys.stderr)

    fit.__doc__ = _sklearn_KNeighborsRegressor.__doc__
    predict.__doc__ = _sklearn_KNeighborsRegressor.predict.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsRegressor.kneighbors.__doc__
    score.__doc__ = _sklearn_KNeighborsRegressor.score.__doc__