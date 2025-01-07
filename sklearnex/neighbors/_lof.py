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

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = _sklearn_LocalOutlierFactor._validate_data


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
    # is hard to unwrap, and this is the most straighforward way.
    @available_if(_sklearn_LocalOutlierFactor._check_novelty_fit_predict)
    @wrap_output_data
    def fit_predict(self, X, y=None):
        """Fit the model to the training set X and return the labels.

        **Not available for novelty detection (when novelty is set to True).**
        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        return self.fit(X)._predict()

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if sklearn_check_version("1.0") and X is not None:
            self._check_feature_names(X, reset=False)
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
    @wrap_output_data
    def score_samples(self, X):
        """Opposite of the Local Outlier Factor of X.

        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.

        **Only available for novelty detection (when novelty is set to True).**
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point. Because of this, the scores obtained via ``score_samples`` may
        differ from the standard LOF scores.
        The standard LOF scores for the training data is available via the
        ``negative_outlier_factor_`` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        opposite_lof_scores : ndarray of shape (n_samples,)
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.
        """
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
