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
from sklearn.decomposition import IncrementalPCA as _sklearn_IncrementalPCA
from sklearn.utils import check_array, gen_batches

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from onedal.decomposition import IncrementalPCA as onedal_IncrementalPCA

from ..._config import get_config
from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain, _add_inc_serialization_note
from ...base import oneDALEstimator
from ...utils.validation import validate_data


@control_n_jobs(
    decorated_methods=["fit", "partial_fit", "transform", "_onedal_finalize_fit"]
)
class IncrementalPCA(oneDALEstimator, _sklearn_IncrementalPCA):

    _need_to_finalize_attrs = {
        "mean_",
        "explained_variance_",
        "explained_variance_ratio_",
        "n_components_",
        "components_",
        "noise_variance_",
        "singular_values_",
        "var_",
    }

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None):
        super().__init__(
            n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size
        )
        self._need_to_finalize = False

    _onedal_incremental_pca = staticmethod(onedal_IncrementalPCA)

    def _onedal_transform(self, X, queue=None):
        assert hasattr(self, "_onedal_estimator")
        if self._need_to_finalize:
            self._onedal_finalize_fit()
        use_raw_input = get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            X = check_array(X, dtype=[np.float64, np.float32])
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_fit_transform(self, X, queue=None):
        self._onedal_fit(X, queue)
        return self._onedal_transform(X, queue)

    def _onedal_partial_fit(self, X, check_input=True, queue=None):
        first_pass = not hasattr(self, "_onedal_estimator")

        use_raw_input = get_config().get("use_raw_input", False) is True
        # never check input when using raw input
        check_input &= use_raw_input is False
        if check_input:
            X = validate_data(self, X, dtype=[np.float64, np.float32], reset=first_pass)

        n_samples, n_features = X.shape

        if self.n_components is None:
            if not hasattr(self, "_components_shape"):
                self.n_components_ = min(n_samples, n_features)
                self._components_shape = self.n_components_

        elif not self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components, n_features)
            )
        elif not self.n_components <= n_samples:
            raise ValueError(
                "n_components=%r must be less or equal to "
                "the batch number of samples "
                "%d." % (self.n_components, n_samples)
            )
        else:
            self.n_components_ = self.n_components

        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = n_samples
        else:
            self.n_samples_seen_ += n_samples

        onedal_params = {"n_components": self.n_components_, "whiten": self.whiten}

        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_pca(**onedal_params)
        self._onedal_estimator.partial_fit(X, queue=queue)
        self._need_to_finalize = True

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

    def _onedal_fit(self, X, queue=None):
        use_raw_input = get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            if sklearn_check_version("1.2"):
                self._validate_params()

            X = validate_data(self, X, dtype=[np.float64, np.float32], copy=self.copy)

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        for batch in gen_batches(n_samples, self.batch_size_):
            X_batch = X[batch]
            self._onedal_partial_fit(X_batch, queue=queue)

        self._onedal_finalize_fit()

        return self

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.decomposition.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def __getattr__(self, attr):
        # finalize the fit if requested attribute requires it
        if attr in IncrementalPCA._need_to_finalize_attrs:
            if "_onedal_estimator" not in self.__dict__:
                # _onedal_estimator required to finalize the fit
                raise AttributeError(
                    f"Requested postfit attribute '{attr}' before fitting the model."
                )
            if self.__dict__["_need_to_finalize"]:
                self._onedal_finalize_fit()
        # join attributes of the class and the onedal_estimator to provide common interface
        joined = self.__dict__ | self.__dict__.get("_onedal_estimator", {}).__dict__
        if attr in joined:
            return joined[attr]
        # raise AttributeError if attribute is neither in this class nor in _onedal_estimator
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

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

    __doc__ = _add_inc_serialization_note(
        _sklearn_IncrementalPCA.__doc__ + "\n" + r"%incremental_serialization_note%"
    )
    fit.__doc__ = _sklearn_IncrementalPCA.fit.__doc__
    fit_transform.__doc__ = _sklearn_IncrementalPCA.fit_transform.__doc__
    transform.__doc__ = _sklearn_IncrementalPCA.transform.__doc__
    partial_fit.__doc__ = _sklearn_IncrementalPCA.partial_fit.__doc__
