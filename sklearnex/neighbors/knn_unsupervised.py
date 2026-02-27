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

from sklearn.neighbors._unsupervised import NearestNeighbors as _sklearn_NearestNeighbors
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal._device_offload import _transfer_to_host
from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors
from onedal.utils._array_api import _is_numpy_namespace

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data
from .common import KNeighborsDispatchingBase


@enable_array_api
@control_n_jobs(decorated_methods=["fit", "kneighbors", "radius_neighbors"])
class NearestNeighbors(KNeighborsDispatchingBase, _sklearn_NearestNeighbors):
    __doc__ = _sklearn_NearestNeighbors.__doc__
    # Default onedal estimator class - SPMD subclasses can override this
    _onedal_estimator = onedal_NearestNeighbors

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
        xp, is_array_api = get_namespace(X)
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
        # Ensure _fit_X matches the input namespace so that
        # kneighbors(X=None) can use get_namespace(self._fit_X).
        if is_array_api and not _is_numpy_namespace(xp):
            device = getattr(X, "device", None)
            self._fit_X = xp.asarray(self._fit_X, device=device)
        return self

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
                "sklearn": _sklearn_NearestNeighbors.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    # radius_neighbors always falls back to sklearn on CPU and returns
    # ragged object-dtype arrays that cannot be converted to array API
    # or GPU formats, so @wrap_output_data is intentionally omitted.
    def radius_neighbors(
        self, X=None, radius=None, return_distance=True, sort_results=False
    ):
        if (
            hasattr(self, "_onedal_estimator")
            or getattr(self, "_tree", 0) is None
            and self._fit_method == "kd_tree"
        ):
            # _fit_X may be on a non-host device (e.g. torch XPU, dpnp GPU)
            # after Array API fit(). Transfer to host for sklearn refit.
            _, (fit_X_host,) = _transfer_to_host(self._fit_X)
            _sklearn_NearestNeighbors.fit(self, fit_X_host, getattr(self, "_y", None))
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
        xp, _ = get_namespace(X)

        if not get_config()["use_raw_input"]:
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
            )

        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        if hasattr(self.__class__, "_onedal_estimator"):
            self._onedal_estimator = self.__class__._onedal_estimator(**onedal_params)
        else:
            self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.fit(X, y, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
                force_all_finite=False,
            )
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Determine if query is the training data
        if X is not None:
            query_is_train = False
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
        # Pass n_neighbors as keyword to avoid _transfer_to_host mixing
        # USM array X with int n_neighbors in the same positional args tuple
        distances, indices = self._onedal_estimator.kneighbors(
            X, n_neighbors=effective_n_neighbors, return_distance=True, queue=queue
        )

        return self._kneighbors_postprocess(
            distances,
            indices,
            n_neighbors if n_neighbors is not None else self.n_neighbors,
            return_distance,
            query_is_train,
        )

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_NearestNeighbors.__doc__
    kneighbors.__doc__ = _sklearn_NearestNeighbors.kneighbors.__doc__
    radius_neighbors.__doc__ = _sklearn_NearestNeighbors.radius_neighbors.__doc__
    radius_neighbors_graph.__doc__ = (
        _sklearn_NearestNeighbors.radius_neighbors_graph.__doc__
    )
