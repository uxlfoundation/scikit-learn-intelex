# ==============================================================================
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
# ==============================================================================

from daal4py.sklearn._utils import get_dtype

from ...datatypes import from_table, to_table
from ...decomposition import IncrementalPCA as base_IncrementalPCA
from ...utils import _check_array
from .._base import BaseEstimatorSPMD


class IncrementalPCA(BaseEstimatorSPMD, base_IncrementalPCA):
    """
    Distributed incremental estimator for PCA based on oneDAL implementation.
    Allows for distributed PCA computation if data is split into batches.

    API is the same as for `onedal.decomposition.IncrementalPCA`
    """

    def _reset(self):
        self._need_to_finalize = False
        self._partial_result = super(base_IncrementalPCA, self)._get_backend(
            "decomposition", "dim_reduction", "partial_train_result"
        )
        if hasattr(self, "components_"):
            del self.components_

    def partial_fit(self, X, y=None, queue=None):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = _check_array(X)
        n_samples, n_features = X.shape

        first_pass = not hasattr(self, "components_")
        if first_pass:
            self.components_ = None
            self.n_samples_seen_ = n_samples
            self.n_features_in_ = n_features
        else:
            self.n_samples_seen_ += n_samples

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        else:
            self.n_components_ = self.n_components

        self._queue = queue

        policy = super(base_IncrementalPCA, self)._get_policy(queue, X)
        X_table = to_table(X, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(X_table)

        self._partial_result = super(base_IncrementalPCA, self)._get_backend(
            "decomposition",
            "dim_reduction",
            "partial_train",
            policy,
            self._params,
            self._partial_result,
            X_table,
        )
        self._need_to_finalize = True
        return self

    def _create_model(self):
        m = super(base_IncrementalPCA, self)._get_backend(
            "decomposition", "dim_reduction", "model"
        )
        m.eigenvectors = to_table(self.components_)
        m.means = to_table(self.mean_)
        if self.whiten:
            m.eigenvalues = to_table(self.explained_variance_)
        self._onedal_model = m
        return m

    def predict(self, X, queue=None):
        policy = super(base_IncrementalPCA, self)._get_policy(queue, X)
        model = self._create_model()
        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X, stage="predict")

        result = super(base_IncrementalPCA, self)._get_backend(
            "decomposition",
            "dim_reduction",
            "infer",
            policy,
            params,
            model,
            X,
        )
        return from_table(result.transformed_data)
