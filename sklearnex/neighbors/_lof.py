#!/usr/bin/env python
# ===============================================================================
# Copyright 2023 Intel Corporation
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
from sklearn.neighbors import LocalOutlierFactor as sklearn_LocalOutlierFactor
from .common import KNeighborsDispatchingBase

try:
    from sklearn.utils.metaestimators import available_if
except ImportError:
    pass

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._utils import control_n_jobs, run_with_n_jobs, sklearn_check_version
from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors

from .._config import config_context
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain


@control_n_jobs
class LocalOutlierFactor(KNeighborsDispatchingBase, sklearn_LocalOutlierFactor):
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **sklearn_LocalOutlierFactor._parameter_constraints
        }

    def __init__(
        self,
        n_neighbors=20,
        *,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        novelty=False,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            contamination=contamination,
            novelty=novelty,
        )

    @run_with_n_jobs
    def _onedal_fit(self, X, y, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = requires_y
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.fit(X, y, queue=queue)

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

        self._distances_fit_X_, _neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_
        )

        # change dtype here?

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

        return self

    def fit(self, X, y=None):
        self._fit_validation(X, y)
        return dispatch(
            self,
            "neighbors.LocalOutlierFactor.fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_LocalOutlierFactor.fit,
            },
            X,
            y,
        )

    @run_with_n_jobs
    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if sklearn_check_version("1.0") and X is not None:
            self._check_feature_names(X, reset=False)
        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": sklearn_NearestNeighbors.kneighbors,
            },
            X,
            n_neighbors,
            return_distance,
        )
