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
        print(f"DEBUG fit START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  y type: {type(y)}, y shape: {getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
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
        
        print(f"DEBUG fit AFTER dispatch:", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        return self

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        
        # CRITICAL FIRST: Ensure _fit_X is always an array before any sklearn operations
        if hasattr(self, '_fit_X') and isinstance(self._fit_X, tuple):
            print("DEBUG kneighbors: PREVENTIVE FIX - _fit_X is tuple, permanently extracting first element", file=sys.stderr)
            self._fit_X = self._fit_X[0]  # Fix the attribute permanently
            
        if X is not None:
            check_feature_names(self, X, reset=False)

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
        print(f"DEBUG radius_neighbors START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  radius: {radius}, return_distance: {return_distance}, sort_results: {sort_results}", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  hasattr _onedal_estimator: {hasattr(self, '_onedal_estimator')}", file=sys.stderr)
        print(f"  _tree: {getattr(self, '_tree', 'NOT_SET')}", file=sys.stderr)
        print(f"  _fit_method: {getattr(self, '_fit_method', 'NOT_SET')}", file=sys.stderr)
        
        # CRITICAL FIRST: Ensure _fit_X is always an array before any sklearn operations
        if hasattr(self, '_fit_X') and isinstance(self._fit_X, tuple):
            print("DEBUG radius_neighbors: PREVENTIVE FIX - _fit_X is tuple, permanently extracting first element", file=sys.stderr)
            self._fit_X = self._fit_X[0]  # Fix the attribute permanently
        
        # Original main branch logic - simple conditional fit
        if (
            hasattr(self, "_onedal_estimator")
            or getattr(self, "_tree", 0) is None
            and self._fit_method == "kd_tree"
        ):
            print("DEBUG: Original condition met - calling sklearn fit", file=sys.stderr)
            print(f"  self._fit_X type before fit: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
            print(f"  self._y type before fit: {type(getattr(self, '_y', 'NOT_SET'))}", file=sys.stderr)
            
            _sklearn_NearestNeighbors.fit(self, self._fit_X, getattr(self, "_y", None))
            print("DEBUG: sklearn fit completed", file=sys.stderr)
        else:
            print("DEBUG: Original condition NOT met - skipping sklearn fit", file=sys.stderr)
        
        check_is_fitted(self)
        
        print(f"DEBUG radius_neighbors BEFORE DISPATCH:", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  radius: {radius}, return_distance: {return_distance}, sort_results: {sort_results}", file=sys.stderr)
        
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
        print(f"DEBUG radius_neighbors_graph START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  radius: {radius}, mode: {mode}, sort_results: {sort_results}", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        # Handle potential tuple in _fit_X before calling dispatch
        if hasattr(self, '_fit_X') and isinstance(self._fit_X, tuple):
            print("DEBUG radius_neighbors_graph: _fit_X is tuple, permanently extracting first element", file=sys.stderr)
            self._fit_X = self._fit_X[0]  # Fix the attribute permanently
            
        print(f"DEBUG radius_neighbors_graph BEFORE DISPATCH:", file=sys.stderr)
        print(f"  self._fit_X type: {type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  radius: {radius}, mode: {mode}, sort_results: {sort_results}", file=sys.stderr)
            
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
        print(f"DEBUG _onedal_fit START - ENTRY PARAMETERS:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  y type: {type(y)}, y shape: {getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  queue: {queue}", file=sys.stderr)
        
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

        print(f"DEBUG _onedal_fit BEFORE calling onedal_estimator.fit:", file=sys.stderr)
        print(f"  X type: {type(X)}, X shape: {getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  y type: {type(y)}, y shape: {getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"  queue: {queue}", file=sys.stderr)

        self._onedal_estimator.fit(X, y, queue=queue)

        print(f"DEBUG _onedal_fit AFTER calling onedal_estimator.fit:", file=sys.stderr)
        print(f"  onedal_estimator._fit_X type: {type(getattr(self._onedal_estimator, '_fit_X', 'NOT_SET'))}", file=sys.stderr)

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
        print(f"DEBUG _save_attributes START:", file=sys.stderr)
        print(f"  onedal_estimator._fit_X type: {type(getattr(self._onedal_estimator, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        if hasattr(self._onedal_estimator, '_fit_X'):
            fit_x_preview = str(self._onedal_estimator._fit_X)[:200]
            print(f"  onedal_estimator._fit_X value preview: {fit_x_preview}", file=sys.stderr)
        
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        fit_x = self._onedal_estimator._fit_X
        
        print(f"DEBUG _save_attributes processing _fit_X:", file=sys.stderr)
        print(f"  fit_x type: {type(fit_x)}", file=sys.stderr)
        print(f"  isinstance(fit_x, tuple): {isinstance(fit_x, tuple)}", file=sys.stderr)
        
        # CRITICAL FIX: OneDAL's to_table() can return tuples (array, None) in recursive calls
        # We must extract the actual array for sklearn compatibility
        if isinstance(fit_x, tuple):
            print(f"DEBUG _save_attributes: fit_x is tuple, extracting array from: {fit_x}", file=sys.stderr)
            self._fit_X = fit_x[0]  # Extract the array from (array, None) tuple
        else:
            self._fit_X = fit_x
        
        print(f"DEBUG _save_attributes AFTER processing:", file=sys.stderr)
        print(f"  self._fit_X type: {type(self._fit_X)}", file=sys.stderr)
        print(f"  self._fit_X shape: {getattr(self._fit_X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_NearestNeighbors.__doc__
    kneighbors.__doc__ = _sklearn_NearestNeighbors.kneighbors.__doc__
    radius_neighbors.__doc__ = _sklearn_NearestNeighbors.radius_neighbors.__doc__
    radius_neighbors_graph.__doc__ = (
        _sklearn_NearestNeighbors.radius_neighbors_graph.__doc__
    )