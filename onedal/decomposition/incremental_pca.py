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

import numpy as np

from daal4py.sklearn._utils import get_dtype

from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_array
from .pca import BasePCA


class IncrementalPCA(BasePCA):

    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method="cov",
        batch_size=None,
        whiten=False,
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic
        self.batch_size = batch_size
        self.whiten = whiten
        module = self._get_backend("decomposition", "dim_reduction")
        self._partial_result = module.partial_train_result()

    def _reset(self):
        module = self._get_backend("decomposition", "dim_reduction")
        del self.components_
        self._partial_result = module.partial_train_result()

    def partial_fit(self, X, queue):
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

        module = self._get_backend("decomposition", "dim_reduction")

        X = _convert_to_supported(self._policy, X)
        
        if not hasattr(self, "_policy"):
            self._policy = self._get_policy(queue, X)

        if not hasattr(self, "_dtype"):
            self._dtype = get_dtype(X)
            self._params = self._get_onedal_params(X)

        X_table = to_table(X)
        self._partial_result = module.partial_train(
            self._policy, self._params, self._partial_result, X_table
        )
        return self

    def finalize_fit(self):
        module = self._get_backend("decomposition", "dim_reduction")
        result = module.finalize_train(self._policy, self._params, self._partial_result)
        self.mean_ = from_table(result.means).ravel()
        self.variances_ = from_table(result.variances)
        self.components_ = from_table(result.eigenvectors)
        self.singular_values_ = np.nan_to_num(from_table(result.singular_values).ravel())
        self.explained_variance_ = np.maximum(from_table(result.eigenvalues).ravel(), 0)
        self.explained_variance_ratio_ = from_table(
            result.explained_variances_ratio
        ).ravel()
        self.noise_variance_ = self._compute_noise_variance(
            self.n_components_, min(self.n_samples_seen_, self.n_features_in_)
        )

        return self
