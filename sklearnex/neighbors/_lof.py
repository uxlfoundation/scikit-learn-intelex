# ===============================================================================
# Copyright 2024 Intel Corporation
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

import warnings
from functools import wraps

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as _sklearn_LocalOutlierFactor
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from sklearnex._device_offload import dispatch, wrap_output_data
from sklearnex.neighbors.common import KNeighborsDispatchingBase
from sklearnex.neighbors.knn_unsupervised import NearestNeighbors

from ..utils._array_api import get_namespace
from ..utils.validation import check_feature_names


@control_n_jobs(decorated_methods=["fit", "kneighbors", "_kneighbors"])
class LocalOutlierFactor(KNeighborsDispatchingBase, _sklearn_LocalOutlierFactor):
    __doc__ = (
        _sklearn_LocalOutlierFactor.__doc__
        + "\n NOTE: When X=None, methods kneighbors, kneighbors_graph, and predict will"
        + "\n only output numpy arrays. In that case, the only way to offload to gpu"
        + "\n is to use a global queue (e.g. using config_context)"
    )
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_LocalOutlierFactor._parameter_constraints
        }

    # Only certain methods should be taken from knn to prevent code
    # duplication. Inheriting would yield a complicated inheritance
    # structure and violate the sklearn inheritance path.
    _save_attributes = NearestNeighbors._save_attributes
    _onedal_knn_fit = NearestNeighbors._onedal_fit
    _onedal_kneighbors = NearestNeighbors._onedal_kneighbors

    def _onedal_fit(self, X, y, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        self._onedal_knn_fit(X, y, queue=queue)

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], " "got: %f" % self.contamination
                )

        n_samples = self.n_samples_fit_

        if self.n_neighbors > n_samples:
            warnings.warn(
                "n_neighbors (%s) is greater than the "
                "total number of samples (%s). n_neighbors "
                "will be set to (n_samples - 1) for estimation."
                % (self.n_neighbors, n_samples)
            )
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        (
            self._distances_fit_X_,
            _neighbors_indices_fit_X_,
        ) = self._onedal_kneighbors(n_neighbors=self.n_neighbors_, queue=queue)

        # Sklearn includes a check for float32 at this point which may not be
        # necessary for onedal

        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_
        )

        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = self._lrd[_neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        if self.contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_, 100.0 * self.contamination
            )

        # adoption of warning for data with duplicated samples from
        # https://github.com/scikit-learn/scikit-learn/pull/28773
        if sklearn_check_version("1.6"):
            if np.min(self.negative_outlier_factor_) < -1e7 and not self.novelty:
                warnings.warn(
                    "Duplicate values are leading to incorrect results. "
                    "Increase the number of neighbors for more accurate results."
                )

        return self

    def fit(self, X, y=None):
        result = dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_LocalOutlierFactor.fit,
            },
            X,
            None,
        )
        return result

    def _predict(self, X=None):
        check_is_fitted(self)

        if X is not None:
            xp, _ = get_namespace(X)
            output = self.decision_function(X) < 0
            is_inlier = xp.ones_like(output, dtype=int)
            is_inlier[output] = -1
        else:
            is_inlier = np.ones(self.n_samples_fit_, dtype=int)
            is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

        return is_inlier

    # This had to be done because predict loses the queue when no
    # argument is given and it is a dpctl tensor or dpnp array.
    # This would cause issues in fit_predict. Also, available_if
    # is hard to unwrap, and this is the most straightforward way.
    @available_if(_sklearn_LocalOutlierFactor._check_novelty_fit_predict)
    @wraps(_sklearn_LocalOutlierFactor.fit_predict, assigned=["__doc__"])
    @wrap_output_data
    def fit_predict(self, X, y=None):
        return self.fit(X)._predict()

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if X is not None:
            check_feature_names(self, X, reset=False)
        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": _sklearn_LocalOutlierFactor.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    kneighbors = wrap_output_data(_kneighbors)

    @available_if(_sklearn_LocalOutlierFactor._check_novelty_score_samples)
    @wraps(_sklearn_LocalOutlierFactor.score_samples, assigned=["__doc__"])
    @wrap_output_data
    def score_samples(self, X):
        check_is_fitted(self)

        distances_X, neighbors_indices_X = self._kneighbors(
            X, n_neighbors=self.n_neighbors_
        )

        X_lrd = self._local_reachability_density(
            distances_X,
            neighbors_indices_X,
        )

        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd[:, np.newaxis]

        return -np.mean(lrd_ratios_array, axis=1)

    fit.__doc__ = _sklearn_LocalOutlierFactor.fit.__doc__
    kneighbors.__doc__ = _sklearn_LocalOutlierFactor.kneighbors.__doc__
