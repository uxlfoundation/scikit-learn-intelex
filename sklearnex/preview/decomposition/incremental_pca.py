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

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA as _sklearn_IncrementalPCA
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.decomposition import IncrementalPCA as onedal_IncrementalPCA

from ..._config import get_config
from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain, _add_inc_serialization_note
from ...base import oneDALEstimator
from ...utils._array_api import get_namespace
from ...utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import StrOptions

@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "transform", "_onedal_finalize_fit"]
)
class IncrementalPCA(oneDALEstimator, _sklearn_IncrementalPCA):
    __doc__ = _sklearn_IncrementalPCA.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_IncrementalPCA._parameter_constraints,
            "svd_solver": [StrOptions({"auto", "covariance_eigh", "onedal_svd"})],
        }

    def __init__(
        self,
        n_components=None,
        *,
        svd_solver="auto",
        whiten=False,
        copy=True,
        batch_size=None,
    ):
        super().__init__(
            n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size
        )
        self.svd_solver = svd_solver
        self._need_to_finalize = False
        # Note: use of the onedal_svd solver will cause partial result to grow proportionally
        # to the input data and for that reason is not the default, which is contrary
        # to the scikit-learn implementation.

    _onedal_incremental_pca = staticmethod(onedal_IncrementalPCA)

    def _onedal_transform(self, X, queue=None):
        # does not batch out data like sklearn's ``IncrementalPCA.transform``
        if self._need_to_finalize:
            self._onedal_finalize_fit()
        if not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(self, X, dtype=[xp.float64, xp.float32], reset=False)
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_fit_transform(self, X, queue=None):
        self._onedal_fit(X, queue=queue)
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_partial_fit(self, X, check_input=True, queue=None):
        first_pass = not hasattr(self, "_onedal_estimator")
        if first_pass:
            self.components_ = None

        if check_input and not get_config()["use_raw_input"]:
                xp, _ = get_namespace(X)
                X = validate_data(self, X, dtype=[xp.float64, xp.float32], reset=first_pass)

        n_samples, n_features = X.shape

        # extracted from sklearn's ``IncrementalPCA.partial_fit``
        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components, n_features)
            )
        elif self.n_components > n_samples and (
            not sklearn_check_version("1.6") or first_pass
        ):
            raise ValueError(
                "n_components=%r must be less or equal to "
                "the batch number of samples "
                "%d." % (self.n_components, n_samples)
                + (
                    "for the first partial_fit call."
                    if sklearn_check_version("1.6")
                    else ""
                )
            )
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (
            self.components_.shape[0] != self.n_components_
        ):
            raise ValueError(
                "Number of input features has changed from %i "
                "to %i between calls to partial_fit! Try "
                "setting n_components to a fixed value."
                % (self.components_.shape[0], self.n_components_)
            )

        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = n_samples
        else:
            self.n_samples_seen_ += n_samples

        onedal_params = {
            "n_components": self.n_components_,
            "whiten": self.whiten,
            "method": "svd" if self.svd_solver == "onedal_svd" else "cov",
        }

        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_pca(**onedal_params)
        self._onedal_estimator.partial_fit(X, queue=queue)
        self._need_to_finalize = True

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()

        # set attributes needed for transform
        self._mean_ = self._onedal_estimator.mean_
        self._components_ = self._onedal_estimator.components_
        self._explained_variance_ = self._onedal_estimator.explained_variance_

        # set other attributes
        self.singular_values_ = self._onedal_estimator.singular_values_
        self.explained_variance_ratio_ = self._onedal_estimator.explained_variance_ratio_
        self.var_ = self._onedal_estimator.var_

        # calculate the noise variance
        xp, _ = get_namespace(self.explained_variance_)
        self.noise_variance_ = xp.mean(self.explained_variance_)
        self._need_to_finalize = False

    def _onedal_fit(self, X, queue=None):
        # Taken from sklearn for conformance purposes
        self.components_ = None

        if not get_config()["use_raw_input"]:
            if sklearn_check_version("1.2"):
                self._validate_params()
            xp, _ = get_namespace(X)
            X = validate_data(self, X, dtype=[xp.float64, xp.float32], copy=self.copy)

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        for batch in gen_batches(
            n_samples, self.batch_size_, min_bin_size=self.n_components or 0
        ):
            X_batch = X[batch, ...]
            self._onedal_partial_fit(X_batch, queue=queue)

        self._onedal_finalize_fit()

        return self

    def _onedal_cpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.decomposition.{self.__class__.__name__}.{method_name}"
        )
        X = data[0]
        if "fit" in method_name:
            patching_status.and_conditions(
                [(not sp.issparse(X), "Sparse input is not supported")]
            )
        else:
            patching_status.and_conditions(
                [(hasattr(self, "_onedal_estimator"), "oneDAL model was not trained")]
            )
        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.decomposition.{self.__class__.__name__}.{method_name}"
        )
        # onedal_svd doesn't exist for GPU
        X = data[0]
        if "fit" in method_name:
            patching_status.and_conditions(
                [
                    (not sp.issparse(X), "Sparse input is not supported"),
                    (self.svd_solver != "onedal_svd", "onedal_svd not supported on GPU"),
                ]
            )
        else:
            patching_status.and_conditions(
                [(hasattr(self, "_onedal_estimator"), "oneDAL model was not trained")]
            )
        return patching_status

    def partial_fit(self, X, y=None, check_input=True):
        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": _sklearn_IncrementalPCA.partial_fit,
            },
            X,
            check_input=check_input,
        )
        return self

    def fit(self, X, y=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_IncrementalPCA.fit,
            },
            X,
        )
        return self

    @wrap_output_data
    def transform(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "transform",
            {
                "onedal": self.__class__._onedal_transform,
                "sklearn": _sklearn_IncrementalPCA.transform,
            },
            X,
        )

    @wrap_output_data
    def fit_transform(self, X, y=None, **fit_params):
        return dispatch(
            self,
            "fit_transform",
            {
                "onedal": self.__class__._onedal_fit_transform,
                "sklearn": _sklearn_IncrementalPCA.fit_transform,
            },
            X,
        )

    # set properties for deleting the onedal_estimator model if:
    # n_components_, components_, means_ or explained_variance_ are
    # changed. This assists in speeding up multiple uses of onedal
    # transform as a model must now only be generated once.

    @property
    def n_components_(self):
        return self._n_components_

    @n_components_.setter
    def n_components_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.n_components_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._n_components_ = value

    @property
    def components_(self):
        return self._components_

    @components_.setter
    def components_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.components_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._n_components_ = value

    @property
    def means_(self):
        return self._means_

    @means_.setter
    def means_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.means_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._means_ = value

    @property
    def explained_variance_(self):
        return self._explained_variance_

    @explained_variance_.setter
    def explained_variance_(self, value):
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.explained_variance_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model
        self._explained_variance_ = value

    __doc__ = _add_inc_serialization_note(
        _sklearn_IncrementalPCA.__doc__ + "\n" + r"%incremental_serialization_note%"
    )
    fit.__doc__ = _sklearn_IncrementalPCA.fit.__doc__
    fit_transform.__doc__ = _sklearn_IncrementalPCA.fit_transform.__doc__
    transform.__doc__ = _sklearn_IncrementalPCA.transform.__doc__
    partial_fit.__doc__ = _sklearn_IncrementalPCA.partial_fit.__doc__
