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

from sklearn.metrics import accuracy_score
from sklearn.neighbors._classification import (
    KNeighborsClassifier as _sklearn_KNeighborsClassifier,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.datatypes import from_table
from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data
from .common import KNeighborsDispatchingBase


@enable_array_api
@control_n_jobs(
    decorated_methods=["fit", "predict", "predict_proba", "kneighbors", "score"]
)
class KNeighborsClassifier(KNeighborsDispatchingBase, _sklearn_KNeighborsClassifier):
    __doc__ = _sklearn_KNeighborsClassifier.__doc__
    # Default onedal estimator class - SPMD subclasses can override this
    _onedal_estimator = onedal_KNeighborsClassifier

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_KNeighborsClassifier._parameter_constraints
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
                "sklearn": _sklearn_KNeighborsClassifier.fit,
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
                "sklearn": _sklearn_KNeighborsClassifier.predict,
            },
            X,
        )

    @wrap_output_data
    def predict_proba(self, X):
        check_is_fitted(self)

        return dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_KNeighborsClassifier.predict_proba,
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
                "sklearn": _sklearn_KNeighborsClassifier.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # Validate n_neighbors parameter first
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
                "sklearn": _sklearn_KNeighborsClassifier.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    def _onedal_fit(self, X, y, queue=None):
        xp, _ = get_namespace(X)

        if not get_config()["use_raw_input"]:
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                multi_output=True,
            )

        # SPMD mode: skip validation but still set effective metric
        self._set_effective_metric()

        # Process classification targets before passing to onedal
        self._process_classification_targets(
            y, skip_validation=get_config()["use_raw_input"]
        )

        # Call onedal backend
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
            self._onedal_estimator = onedal_KNeighborsClassifier(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.classes_ = self.classes_
        self._onedal_estimator._y = self._y
        self._onedal_estimator.outputs_2d_ = self.outputs_2d_
        self._onedal_estimator._shape = self._shape

        self._onedal_estimator.fit(X, y, queue=queue)

        # Post-processing
        self._save_attributes()

    def _process_classification_targets(self, y, skip_validation=False):
        """Process classification targets and set class-related attributes.

        Parameters
        ----------
        y : array-like
            Target values
        skip_validation : bool, default=False
            If True, skip check_classification_targets validation.
            Used when use_raw_input=True (SPMD mode).
        """
        # Array API support: get namespace from y
        xp, _ = get_namespace(y)

        # y should already be numpy array from validate_data
        y = xp.asarray(y)

        # Handle shape processing
        shape = getattr(y, "shape", None)
        self._shape = shape if shape is not None else y.shape

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            self.outputs_2d_ = False
            y = xp.reshape(y, (-1, 1))
        else:
            self.outputs_2d_ = True

        # Validate classification targets (skipped only for use_raw_input/SPMD mode).
        if not skip_validation:
            check_classification_targets(y)

        # Process classes using unique_inverse (numpy 2.0+ and Array API)
        # or unique with return_inverse (older numpy)
        n_outputs = y.shape[1]
        self.classes_ = [None] * n_outputs
        self._y = xp.empty_like(y, dtype=xp.int64)
        for k in range(n_outputs):
            if hasattr(xp, "unique_inverse"):
                classes_k, inverse_k = xp.unique_inverse(y[:, k])
            else:
                classes_k, inverse_k = xp.unique(y[:, k], return_inverse=True)
            n_classes = classes_k.shape[0]
            if n_classes > xp.iinfo(xp.int64).max:
                raise ValueError(
                    f"Number of classes ({n_classes}) exceeds int64 dtype limit."
                )
            self.classes_[k] = classes_k
            self._y[:, k] = xp.asarray(inverse_k, dtype=xp.int64)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = xp.reshape(self._y, (-1,))

        # Validate we have at least 2 classes
        self._validate_n_classes()

    def _onedal_predict(self, X, queue=None):
        if X is not None and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        params = self._onedal_estimator._get_onedal_params(X)
        params["result_option"] = "responses"
        result = self._onedal_estimator._onedal_predict(
            self._onedal_estimator._onedal_model, X, params
        )
        xp, _ = get_namespace(X)
        responses = from_table(result.responses, like=X)
        return self.classes_.take(xp.asarray(responses.ravel(), dtype=xp.int64))

    def _onedal_predict_proba(self, X, queue=None):
        if X is not None and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        neigh_dist, neigh_ind = self._onedal_estimator.kneighbors(X)

        return self._compute_class_probabilities(
            neigh_dist, neigh_ind, self.weights, self._y, self.classes_, self.outputs_2d_
        )

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Only skip validation when use_raw_input=True (SPMD mode)
        use_raw_input = get_config()["use_raw_input"]

        if X is not None and not use_raw_input:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._y = self._onedal_estimator._y
        self._fit_method = self._onedal_estimator._fit_method
        self.outputs_2d_ = self._onedal_estimator.outputs_2d_
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_KNeighborsClassifier.fit.__doc__
    predict.__doc__ = _sklearn_KNeighborsClassifier.predict.__doc__
    predict_proba.__doc__ = _sklearn_KNeighborsClassifier.predict_proba.__doc__
    score.__doc__ = _sklearn_KNeighborsClassifier.score.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsClassifier.kneighbors.__doc__
