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

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors._classification import (
    KNeighborsClassifier as _sklearn_KNeighborsClassifier,
)
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier

from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase


@enable_array_api
@control_n_jobs(
    decorated_methods=["fit", "predict", "predict_proba", "kneighbors", "score"]
)
class KNeighborsClassifier(KNeighborsDispatchingBase, _sklearn_KNeighborsClassifier):
    __doc__ = _sklearn_KNeighborsClassifier.__doc__
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

        result = dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_KNeighborsClassifier.predict,
            },
            X,
        )
        return result

    @wrap_output_data
    def predict_proba(self, X):
        check_is_fitted(self)

        result = dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_KNeighborsClassifier.predict_proba,
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
                "sklearn": _sklearn_KNeighborsClassifier.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )
        return result

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # Validate n_neighbors parameter first
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
                "sklearn": _sklearn_KNeighborsClassifier.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )
        return result

    def _onedal_fit(self, X, y, queue=None):
        xp, _ = get_namespace(X)
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[xp.float64, xp.float32],
            accept_sparse="csr",
        )
        # Process classification targets in sklearnex before passing to onedal
        self._process_classification_targets(y)
        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        self._onedal_estimator = onedal_KNeighborsClassifier(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_

        # Pass both original and processed targets to onedal
        # onedal needs the processed classes_ and _y attributes that we just set
        self._onedal_estimator.classes_ = self.classes_
        self._onedal_estimator._y = self._y
        self._onedal_estimator.outputs_2d_ = self.outputs_2d_
        self._onedal_estimator._shape = self._shape  # Pass shape from sklearnex

        # Pass original y to onedal - it will use the pre-set classes_ and _y attributes we just assigned
        self._onedal_estimator.fit(X, y, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        # Use the unified helper from common.py (calls kneighbors + computes prediction)
        # This properly handles X=None (LOOCV) case
        result = self._predict_skl_classification(X)
        return result

    def _onedal_predict_proba(self, X, queue=None):
        # Call kneighbors through sklearnex (self.kneighbors is the sklearnex method)
        # This properly handles X=None case (LOOCV) with query_is_train logic
        neigh_dist, neigh_ind = self.kneighbors(X)

        # Use the helper method to compute class probabilities
        result = self._compute_class_probabilities(
            neigh_dist, neigh_ind, self.weights, self._y, self.classes_, self.outputs_2d_
        )
        return result

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        if X is not None:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

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
        # Convert array API to numpy for sklearn's accuracy_score
        # Note: validate_data does NOT convert array API to numpy, so we do it explicitly
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        result = accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )
        return result

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
