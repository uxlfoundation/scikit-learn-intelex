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
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase


@enable_array_api
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
        print(f"DEBUG NearestNeighbors.fit START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}, y type={type(y)}", file=sys.stderr)
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
        print(f"DEBUG NearestNeighbors.fit END: _fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}, _fit_X shape={getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        return self

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        print(f"DEBUG NearestNeighbors.kneighbors START: X type={type(X)}, _fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        
        # Validate n_neighbors parameter first (before check_is_fitted)
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)
        
        check_is_fitted(self)
        if X is not None:
            check_feature_names(self, X, reset=False)
        
        # Validate kneighbors parameters (inherited from KNeighborsDispatchingBase)
        self._kneighbors_validation(X, n_neighbors)
        
        result = dispatch(
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
        print(f"DEBUG NearestNeighbors.kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    @wrap_output_data
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        print(f"DEBUG NearestNeighbors.radius_neighbors START: X type={type(X)}, _fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}, _fit_X shape={getattr(getattr(self, '_fit_X', None), 'shape', 'NO_SHAPE')}", file=sys.stderr)
        print(f"DEBUG radius_neighbors: hasattr _onedal_estimator={hasattr(self, '_onedal_estimator')}, _tree={getattr(self, '_tree', 'NOT_SET')}, _fit_method={getattr(self, '_fit_method', 'NOT_SET')}", file=sys.stderr)
        if (
            hasattr(self, "_onedal_estimator")
            or getattr(self, "_tree", 0) is None
            and self._fit_method == "kd_tree"
        ):
            print(f"DEBUG radius_neighbors: Calling sklearn fit with _fit_X type={type(self._fit_X)}", file=sys.stderr)
            _sklearn_NearestNeighbors.fit(self, self._fit_X, getattr(self, "_y", None))
            print(f"DEBUG radius_neighbors: sklearn fit completed, _fit_X type now={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        check_is_fitted(self)
        result = dispatch(
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
        print(f"DEBUG NearestNeighbors.radius_neighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def radius_neighbors_graph(
        self, X=None, radius=None, mode="connectivity", sort_results=False
    ):
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
        print(f"DEBUG NearestNeighbors._onedal_fit START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}, y type={type(y)}", file=sys.stderr)
        
        # Get array namespace for array API support
        xp, _ = get_namespace(X)
        print(f"DEBUG: Array namespace: {xp}", file=sys.stderr)
        
        # REFACTOR: Use validate_data from sklearnex.utils.validation to convert pandas to numpy
        X = validate_data(
            self, X, dtype=[xp.float64, xp.float32], accept_sparse="csr"
        )
        print(f"DEBUG: After validate_data, X type={type(X)}", file=sys.stderr)
        
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
        print(f"DEBUG NearestNeighbors._onedal_fit: Calling onedal_estimator.fit", file=sys.stderr)
        self._onedal_estimator.fit(X, y, queue=queue)
        print(f"DEBUG NearestNeighbors._onedal_fit: After fit, onedal_estimator._fit_X type={type(getattr(self._onedal_estimator, '_fit_X', 'NOT_SET'))}", file=sys.stderr)

        self._save_attributes()
        print(f"DEBUG NearestNeighbors._onedal_fit END: self._fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)

    def _onedal_predict(self, X, queue=None):
        # Validate and convert X (pandas to numpy if needed) only if X is not None
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self, X, dtype=[xp.float64, xp.float32], accept_sparse="csr", reset=False
            )
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        import sys
        print(f"DEBUG NearestNeighbors._onedal_kneighbors START: X type={type(X)}, n_neighbors={n_neighbors}, return_distance={return_distance}", file=sys.stderr)
        
        # REFACTOR: All post-processing now in sklearnex following PCA pattern
        # Prepare inputs and handle query_is_train case (includes validation AFTER +=1)
        X, n_neighbors, query_is_train = self._prepare_kneighbors_inputs(X, n_neighbors)
        
        # Get raw results from onedal backend
        result = self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )
        
        # Apply post-processing (kd_tree sorting, removing self from results)
        result = self._kneighbors_post_processing(X, n_neighbors, return_distance, result, query_is_train)
        
        print(f"DEBUG NearestNeighbors._onedal_kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def _save_attributes(self):
        print(f"DEBUG NearestNeighbors._save_attributes START: onedal_estimator._fit_X type={type(getattr(self._onedal_estimator, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        if hasattr(self._onedal_estimator, '_fit_X'):
            fit_x_preview = str(self._onedal_estimator._fit_X)[:200]
            print(f"DEBUG _save_attributes: _fit_X value preview={fit_x_preview}", file=sys.stderr)
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        # ORIGINAL MAIN BRANCH: Direct assignment without any tuple extraction
        self._fit_X = self._onedal_estimator._fit_X
        print(f"DEBUG _save_attributes: AFTER assignment - self._fit_X type={type(self._fit_X)}, has shape attr={hasattr(self._fit_X, 'shape')}", file=sys.stderr)
        if hasattr(self._fit_X, 'shape'):
            print(f"DEBUG _save_attributes: self._fit_X.shape={self._fit_X.shape}", file=sys.stderr)
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree
        print(f"DEBUG NearestNeighbors._save_attributes END: _fit_method={self._fit_method}, _tree={self._tree}", file=sys.stderr)

    fit.__doc__ = _sklearn_NearestNeighbors.__doc__
    kneighbors.__doc__ = _sklearn_NearestNeighbors.kneighbors.__doc__
    radius_neighbors.__doc__ = _sklearn_NearestNeighbors.radius_neighbors.__doc__
    radius_neighbors_graph.__doc__ = (
        _sklearn_NearestNeighbors.radius_neighbors_graph.__doc__
    )