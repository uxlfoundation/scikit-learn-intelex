# ===============================================================================
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
# ===============================================================================

import sys
import numpy as np
from sklearn.neighbors._unsupervised import NearestNeighbors as _sklearn_NearestNeighbors
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors

from .._device_offload import dispatch, wrap_output_data
from ..utils.validation import check_feature_names
from .common import KNeighborsDispatchingBase


@control_n_jobs(decorated_methods=["fit", "kneighbors", "radius_neighbors"])
class NearestNeighbors(KNeighborsDispatchingBase, _sklearn_NearestNeighbors):
    __doc__ = _sklearn_NearestNeighbors.__doc__
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_NearestNeighbors._parameter_constraints
        }

    @_deprecate_positional_args
    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
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

    def fit(self, X, y=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_NearestNeighbors.fit,
            },
            X,
            None,
        )
        return self

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if X is not None:
            check_feature_names(self, X, reset=False)
            # Perform preprocessing at sklearnex level
            from onedal.utils.validation import _check_array

            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
            self._validate_feature_count(X, "kneighbors")

        # Validate n_neighbors
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)

        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": _sklearn_NearestNeighbors.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    @wrap_output_data
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        print(f"DEBUG radius_neighbors start: hasattr _onedal_estimator: {hasattr(self, '_onedal_estimator')}", file=sys.stderr)
        print(f"DEBUG radius_neighbors start: _tree: {getattr(self, '_tree', 'NOT_SET')}", file=sys.stderr)
        print(f"DEBUG radius_neighbors start: _fit_method: {getattr(self, '_fit_method', 'NOT_SET')}", file=sys.stderr)
        
        # Check the condition logic
        has_onedal = hasattr(self, "_onedal_estimator")
        tree_is_none = getattr(self, "_tree", 0) is None
        is_kd_tree = getattr(self, "_fit_method", None) == "kd_tree"
        print(f"DEBUG: has_onedal={has_onedal}, tree_is_none={tree_is_none}, is_kd_tree={is_kd_tree}", file=sys.stderr)
        
        condition_met = has_onedal or (tree_is_none and is_kd_tree)
        print(f"DEBUG: condition_met={condition_met}", file=sys.stderr)
        
        if condition_met:
            print("DEBUG: Entering the fit_x handling block", file=sys.stderr)
            # Handle potential tuple in _fit_X (same as _save_attributes logic)
            fit_x = self._fit_X
            print(f"DEBUG radius_neighbors: _fit_X type: {type(fit_x)}", file=sys.stderr)
            print(f"DEBUG radius_neighbors: _fit_X shape/content: {fit_x.shape if hasattr(fit_x, 'shape') else fit_x}", file=sys.stderr)
            fit_x_array = fit_x[0] if isinstance(fit_x, tuple) else fit_x
            print(f"DEBUG radius_neighbors: fit_x_array type: {type(fit_x_array)}", file=sys.stderr)
            
            # Additional safety check - ensure fit_x_array is not a tuple
            if isinstance(fit_x_array, tuple):
                print(f"DEBUG radius_neighbors: fit_x_array is still tuple after extraction: {type(fit_x_array)}", file=sys.stderr)
                fit_x_array = fit_x_array[0]  # Extract again if needed
                print(f"DEBUG radius_neighbors: fit_x_array after second extraction: {type(fit_x_array)}", file=sys.stderr)
            
            # Temporarily set _fit_X to the extracted array since sklearn accesses it directly
            original_fit_x = self._fit_X
            self._fit_X = fit_x_array
            
            # Debug the _y value and handle potential tuple
            y_value = getattr(self, "_y", None)
            if isinstance(y_value, tuple):
                print(f"DEBUG: _y is tuple, extracting: {type(y_value)}", file=sys.stderr)
                y_value = y_value[0] if y_value[0] is not None else None
            print(f"DEBUG: _y value type: {type(y_value)}, value: {y_value}", file=sys.stderr)
            
            try:
                # Call _fit directly to avoid any preprocessing in fit() that might create tuples
                _sklearn_NearestNeighbors._fit(self, fit_x_array, y_value)
            finally:
                # Restore original _fit_X
                self._fit_X = original_fit_x
        else:
            print("DEBUG: NOT entering the fit_x handling block - using default path", file=sys.stderr)
            # ALWAYS handle potential tuple in _fit_X for robustness
            if hasattr(self, '_fit_X'):
                fit_x = self._fit_X
                print(f"DEBUG fallback path: _fit_X type: {type(fit_x)}", file=sys.stderr)
                if isinstance(fit_x, tuple):
                    print("DEBUG fallback path: _fit_X is tuple, extracting first element", file=sys.stderr)
                    self._fit_X = fit_x[0]
        check_is_fitted(self)
        return dispatch(
            self,
            "radius_neighbors",
            {
                "onedal": None,
                "sklearn": _sklearn_NearestNeighbors.radius_neighbors,
            },
            X,
            radius=radius,
            return_distance=return_distance,
            sort_results=sort_results,
        )

    def radius_neighbors_graph(
        self, X=None, radius=None, mode="connectivity", sort_results=False
    ):
        print(f"DEBUG radius_neighbors_graph start: _fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        # Handle potential tuple in _fit_X before calling dispatch
        if hasattr(self, '_fit_X') and isinstance(self._fit_X, tuple):
            print("DEBUG radius_neighbors_graph: _fit_X is tuple, extracting first element", file=sys.stderr)
            self._fit_X = self._fit_X[0]
            
        return dispatch(
            self,
            "radius_neighbors_graph",
            {
                "onedal": None,
                "sklearn": _sklearn_NearestNeighbors.radius_neighbors_graph,
            },
            X,
            radius=radius,
            mode=mode,
            sort_results=sort_results,
        )

    def _onedal_fit(self, X, y=None, queue=None):
        # Perform preprocessing at sklearnex level
        X, _ = self._validate_data(X, dtype=[np.float64, np.float32], accept_sparse=True)

        # Validate n_neighbors
        self._validate_n_neighbors(self.n_neighbors)

        # Parse auto method
        self._fit_method = self._parse_auto_method(self.algorithm, X.shape[0], X.shape[1])

        # Set basic attributes for unsupervised
        self.classes_ = None

        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator._fit_method = self._fit_method

        # Set attributes on the onedal estimator
        self._onedal_estimator.classes_ = self.classes_

        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    def _save_attributes(self):
        print(f"DEBUG: _save_attributes - _fit_X type: {type(self._onedal_estimator._fit_X)}", file=sys.stderr)
        if hasattr(self._onedal_estimator, '_fit_X'):
            print(f"DEBUG: _fit_X value preview: {str(self._onedal_estimator._fit_X)[:200]}", file=sys.stderr)
        
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        fit_x = self._onedal_estimator._fit_X
        self._fit_X = fit_x[0] if isinstance(fit_x, tuple) else fit_x
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_NearestNeighbors.__doc__
    kneighbors.__doc__ = _sklearn_NearestNeighbors.kneighbors.__doc__
    radius_neighbors.__doc__ = _sklearn_NearestNeighbors.radius_neighbors.__doc__
    radius_neighbors_graph.__doc__ = (
        _sklearn_NearestNeighbors.radius_neighbors_graph.__doc__
    )